"""
neo4j_helpers.py
----------------
Shared Neo4j connection and schema-introspection utilities.
All other modules import from here — nothing in this file touches the LLM.
"""
from _future_ import annotations

import logging
from collections import defaultdict

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_driver():
    """Create and verify a Neo4j driver. Raises on connection failure."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


# ---------------------------------------------------------------------------
# Schema introspection  (no data scanning)
# ---------------------------------------------------------------------------

def get_labels(driver) -> list[str]:
    """Return every node label present in the graph."""
    with driver.session(database=NEO4J_DATABASE) as s:
        return [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]


def get_schema(driver) -> str:
    """
    Return the relationship schema as human-readable lines.
    Uses db.schema.visualization() — no data scanning.
    Example line:  COUNTRY -[EXPORTS]-> ACTIVE_INGREDIENT
    """
    cypher = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        UNWIND relationships AS r
        RETURN labels(startNode(r))[0] AS from_label,
               type(r)                AS rel,
               labels(endNode(r))[0]  AS to_label
        ORDER BY from_label, rel
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        rows  = list(s.run(cypher))
        lines = [
            f"{r['from_label']} -[{r['rel']}]-> {r['to_label']}"
            for r in rows
            if r["from_label"] and r["to_label"]
        ]
    return "\n".join(lines) if lines else "No relationships found."


def get_node_properties(driver) -> str:
    """
    Return the property catalogue for every label.
    Uses db.schema.nodeTypeProperties() — no data scanning.
    Example line:  BRAND                    : name (STRING), category (STRING)
    """
    cypher = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeLabels, propertyName, propertyTypes
        RETURN nodeLabels, propertyName, propertyTypes
        ORDER BY nodeLabels, propertyName
    """
    props: dict[str, list[str]] = defaultdict(list)
    with driver.session(database=NEO4J_DATABASE) as s:
        for row in s.run(cypher):
            label     = row["nodeLabels"][0] if row["nodeLabels"] else "UNKNOWN"
            prop_name = row["propertyName"] or ""
            prop_type = ", ".join(row["propertyTypes"]) if row["propertyTypes"] else "UNKNOWN"
            if prop_name:
                props[label].append(f"{prop_name} ({prop_type})")

    if not props:
        return "No node properties found."

    return "\n".join(
        f"{label:<25}: {', '.join(prop_list)}"
        for label, prop_list in sorted(props.items())
    )


# ---------------------------------------------------------------------------
# Relationship property introspection
# ---------------------------------------------------------------------------

def get_relationship_properties(driver) -> str:
    """
    Return the property catalogue for every relationship type.
    Uses db.schema.relTypeProperties() — no data scanning.
    Example line:  EXPORTS                  : id (STRING), volume (FLOAT)
    """
    cypher = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        RETURN relType, propertyName, propertyTypes
        ORDER BY relType, propertyName
    """
    props: dict[str, list[str]] = defaultdict(list)
    with driver.session(database=NEO4J_DATABASE) as s:
        for row in s.run(cypher):
            # relType comes back as ":REL_NAME" in some Neo4j versions — strip punctuation
            raw_type  = row["relType"] or ""
            rel_type  = raw_type.strip().strip(":`")
            prop_name = row["propertyName"] or ""
            prop_type = ", ".join(row["propertyTypes"]) if row["propertyTypes"] else "UNKNOWN"
            if prop_name:
                props[rel_type].append(f"{prop_name} ({prop_type})")

    if not props:
        return "No relationship properties found."

    return "\n".join(
        f"{rel_type:<25}: {', '.join(prop_list)}"
        for rel_type, prop_list in sorted(props.items())
    )


# ---------------------------------------------------------------------------
# Filtering helpers (used by downstream stages)
# ---------------------------------------------------------------------------

def filter_relationship_properties(rel_properties: str, schema: str, label_names: list[str]) -> str:
    """
    Keep only relationship property lines whose relationship type appears
    in schema lines involving the given labels.
    """
    import re
    relevant_rels: set[str] = set()
    for line in schema.splitlines():
        if any(lbl in line for lbl in label_names):
            match = re.search(r"\[([A-Z_]+)\]", line)
            if match:
                relevant_rels.add(match.group(1))

    if not relevant_rels:
        return rel_properties  # can't filter, return all

    lines = [
        line for line in rel_properties.splitlines()
        if any(line.strip().startswith(rel) for rel in relevant_rels)
    ]
    return "\n".join(lines) if lines else rel_properties


def filter_schema(schema: str, label_names: list[str]) -> str:
    """Keep only schema lines that involve at least one of the given labels."""
    label_set = set(label_names)
    lines = [
        line for line in schema.splitlines()
        if any(lbl in line for lbl in label_set)
    ]
    return "\n".join(lines) if lines else schema


def filter_node_properties(node_properties: str, label_names: list[str]) -> str:
    """Keep only property lines whose label is in the given set."""
    label_set = set(label_names)
    lines = [
        line for line in node_properties.splitlines()
        if any(line.strip().startswith(lbl) for lbl in label_set)
    ]
    return "\n".join(lines) if lines else node_properties


# ---------------------------------------------------------------------------
# Commodity symbol resolution (full-text fuzzy matching)
# ---------------------------------------------------------------------------

FULLTEXT_INDEX_NAME = "commodity_search"


def ensure_commodity_fulltext_index(driver) -> None:
    """
    Idempotently create the full-text index on COMMODITY nodes.
    Safe to call on every startup — Neo4j will no-op if the index exists.
    """
    cypher = (
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
        "FOR (c:COMMODITY) ON EACH [c.name, c.aliases]"
    )
    with driver.session(database=NEO4J_DATABASE) as s:
        s.run(cypher)
    logger.info("Ensured full-text index '%s' exists.", FULLTEXT_INDEX_NAME)


def resolve_commodities_fuzzy(
    driver,
    mentions: list[str],
    *,
    score_threshold: float = 0.5,
) -> dict[str, str | None]:
    """
    Fuzzy-match a list of commodity mentions against the Neo4j full-text index
    and return a mapping of {mention: symbol} for each resolved commodity.

    Uses Lucene fuzzy syntax ("term~") to handle typos and partial matches.
    Only returns a match if the full-text score exceeds score_threshold.

    Args:
        driver      : Active Neo4j driver instance.
        mentions    : Raw commodity mentions extracted by the LLM.
        score_threshold : Minimum Lucene score to accept a match (default 0.5).

    Returns:
        {"wheat": "BL1:COM", "soybean": "S 1:COM", "xyz": None}
        None values indicate unresolved mentions.
    """
    if not mentions:
        return {}

    resolved: dict[str, str | None] = {}

    with driver.session(database=NEO4J_DATABASE) as s:
        for mention in mentions:
            # Lucene fuzzy query: append ~ for edit-distance matching
            # Escape special chars and split multi-word mentions
            terms = mention.strip().split()
            lucene_query = " ".join(f"{t}~" for t in terms if t)

            if not lucene_query:
                resolved[mention] = None
                continue

            try:
                result = s.run(
                    """
                    CALL db.index.fulltext.queryNodes($index, $query)
                    YIELD node, score
                    WHERE score > $threshold
                    RETURN node.name AS name, node.symbol AS symbol, score
                    ORDER BY score DESC
                    LIMIT 1
                    """,
                    index=FULLTEXT_INDEX_NAME,
                    query=lucene_query,
                    threshold=score_threshold,
                )
                record = result.single()
                if record and record["symbol"]:
                    resolved[mention] = record["symbol"]
                    logger.info(
                        "Commodity resolved: '%s' -> '%s' (score=%.3f)",
                        mention, record["symbol"], record["score"],
                    )
                else:
                    resolved[mention] = None
                    logger.warning(
                        "Commodity unresolved: '%s' — no match above threshold.",
                        mention,
                    )
            except Exception as exc:
                logger.error("Fuzzy lookup failed for '%s': %s", mention, exc)
                resolved[mention] = None

    return resolved