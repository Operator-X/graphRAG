"""
cypher_gen.py  --  Stage 4
--------------------------
Generates a Cypher READ query from the user query + ranked labels (Stage 3
output), executes it against Neo4j, and returns the raw rows.

Input  : query, ranked_labels (list[dict]), schema, node_properties, driver,
         resolved_entities (dict from Stage 2, optional)
Output : dict  -- {"cypher": str, "explanation": str, "results": list[dict]}
"""
from _future_ import annotations

import json
import logging
import re
import time

from openai import AzureOpenAI, RateLimitError, APIConnectionError, APIError

from config import (
    AZURE_KEY, AZURE_VERSION, AZURE_ENDPOINT, AZURE_MODEL,
    NEO4J_DATABASE, CYPHER_DEFAULT_LIMIT, CYPHER_MAX_RETRIES,
)
from neo4j_helpers import filter_schema, filter_node_properties, filter_relationship_properties

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety guard -- reject any query that tries to write to the graph
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL\s+apoc\..*write)\b",
    re.IGNORECASE,
)

def _is_safe_read(cypher: str) -> bool:
    return not bool(_WRITE_KEYWORDS.search(cypher))


# ---------------------------------------------------------------------------
# Prompt sections -- assembled conditionally based on ranked labels
# ---------------------------------------------------------------------------

_PROMPT_BASE = """\
You are a Neo4j Cypher expert for UPL's internal macroeconomic knowledge graph.

Domain context:
  ACTIVE_INGREDIENT is the central entity. It represents the chemical compounds
  in crop-protection products. Connected node types:
    - BRAND    : UPL commercial products containing active ingredients.
    - COMMODITY: crops/raw materials linked to active ingredients via production
                 or usage relationships.
    - COUNTRY  : nations that export or import active ingredients and commodities.
    - CURRENCY : currencies used by countries in trade.

Your task:
  Given a user query and the relevant labels (ranked by importance), write a
  single valid Cypher READ query that answers the question.

  Return ONLY a JSON object with:
    - "cypher"      : the complete Cypher query string
    - "explanation" : one sentence describing what the query does

Rules:
  1. Use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
     Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, or DROP.
  2. Always add LIMIT (use 25 unless the question clearly needs more or fewer).
  3. Use only the label names, relationship types, and property names visible
     in the provided schema and node-property catalogue.
  4. Prefer returning specific named properties (e.g. a.name) over entire nodes.
  5. Use toLower() for case-insensitive string matching when filtering by name.
  6. If the question asks for a count or aggregation, use COUNT / SUM / AVG.
  7. Start traversals from the most specific node (highest-ranked label) and
     traverse outward following the schema relationships.
   8. RELATIONSHIP PROPERTIES -- When "Relationship properties" are provided,
      use them for filtering or returning attributes of relationships. For
      example, if EXPORTS has a volume property, you can filter or aggregate
      on it (e.g. ORDER BY e.volume DESC, or RETURN sum(e.volume) AS total).
      Never invent relationship property names -- only use those listed.
"""

# ---------------------------------------------------------------------------
# Entity matching rules (always included -- lightweight)
# ---------------------------------------------------------------------------

_PROMPT_MATCHING = """\
  COUNTRY MATCHING RULE -- When "Resolved entities" provides ISO codes for countries
  mentioned in the query, always match those countries using the iso_code property:
      WHERE c.iso_code = '<code>'
  instead of matching on the name property. This is mandatory -- it handles aliases
  like "USA" -> "US", "Britain" -> "GB", "Emirates" -> "AE", etc. Only fall back to
  toLower(c.name) CONTAINS matching if NO iso_code is provided for that country.

  ACTIVE INGREDIENT MATCHING RULE -- When "Resolved entities" provides names under
  the "active_ingredients" key, use the resolved (title-cased) name for matching:
      WHERE toLower(ai.name) = toLower('<resolved_name>')
  This is more precise than CONTAINS. Fall back to CONTAINS only if no resolved
  name is provided for that ingredient.
  CATEGORY TERMS: Words like 'pesticides', 'herbicides', 'fungicides',
  'insecticides', 'agrochemicals' are category/type names, NOT active ingredient
  names. When the query contains only a category term and no resolved ingredient
  is provided under "active_ingredients":
    a. Do NOT filter on ai.name at all for category-only queries — return all
       active ingredients linked via the relationship.
    b. If the graph stores a 'type' property on ACTIVE_INGREDIENT, you MAY
       additionally filter with: WHERE toLower(ai.type) CONTAINS '<category>'
    c. NEVER write WHERE toLower(ai.name) CONTAINS 'pesticide' or similar —
       no active ingredient is named 'pesticide'.
"""

# ---------------------------------------------------------------------------
# Trade corridor section (included when COUNTRY + ACTIVE_INGREDIENT present)
# ---------------------------------------------------------------------------

_PROMPT_TRADE = """\
  CRITICAL -- Trade corridor structure:
    Import and export relationships are NOT simple 2-node links. Every trade flow
    is a 3-hop corridor:

        (origin:COUNTRY) -[e:EXPORTS {id: <corridor_id>}]-> (ai:ACTIVE_INGREDIENT)
                         -[t:TO      {id: <corridor_id>}]-> (dest:COUNTRY)

    The EXPORTS (or IMPORTS) relationship and the TO relationship share the same
    id value. You MUST always include the constraint e.id = t.id (or i.id = t.id
    for IMPORTS) to correctly identify a single trade corridor.

    Semantics:
      - The COUNTRY at the start of EXPORTS / IMPORTS is the EXPORTING country (origin).
      - The COUNTRY at the end of TO is the IMPORTING country (destination).

    NEVER write a simple 2-hop query such as:
        MATCH (c:COUNTRY)-[:EXPORTS]->(ai:ACTIVE_INGREDIENT)
    for any trade, import, or export question -- it ignores the destination country
    and violates the corridor identity constraint.

  TRADE CORRIDOR RULE -- For ANY query involving imports, exports, trade flows,
  or country-to-country movement of active ingredients:
    a. Always use the full 3-hop corridor path:
           (origin:COUNTRY)-[e:EXPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY)
       or for IMPORTS:
           (origin:COUNTRY)-[i:IMPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY)
    b. Always include the corridor id constraint: WHERE e.id = t.id  (or i.id = t.id)
    c. Always return both origin and destination country so the full corridor is visible.

Trade corridor examples:

  Query   : "which countries export Glyphosate?"
  Output  : {
    "cypher": "MATCH (origin:COUNTRY)-[e:EXPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY) WHERE e.id = t.id AND toLower(ai.name) CONTAINS 'glyphosate' RETURN origin.name AS exporting_country, dest.name AS importing_country, ai.name AS ingredient LIMIT 25",
    "explanation": "Finds all trade corridors where a country exports Glyphosate to another country, enforcing the corridor id constraint."
  }

  Query   : "which countries import Glyphosate?"
  Output  : {
    "cypher": "MATCH (origin:COUNTRY)-[e:EXPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY) WHERE e.id = t.id AND toLower(ai.name) CONTAINS 'glyphosate' RETURN dest.name AS importing_country, origin.name AS exporting_country, ai.name AS ingredient LIMIT 25",
    "explanation": "Finds all Glyphosate trade corridors and returns the destination (importing) countries along with their source countries."
  }

  Query   : "which active ingredients does the USA export to India?"
  Resolved entities: {"USA": "US", "India": "IN"}
  Output  : {
    "cypher": "MATCH (origin:COUNTRY)-[e:EXPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY) WHERE e.id = t.id AND origin.iso_code = 'US' AND dest.iso_code = 'IN' RETURN ai.name AS ingredient, origin.name AS exporting_country, dest.name AS importing_country LIMIT 25",
    "explanation": "Finds all active ingredients exported from the US to India using ISO code matching on both countries."
  }

  Query   : "what does UK import from Brazil?"
  Resolved entities: {"UK": "GB", "Brazil": "BR"}
  Output  : {
    "cypher": "MATCH (origin:COUNTRY)-[e:EXPORTS]->(ai:ACTIVE_INGREDIENT)-[t:TO]->(dest:COUNTRY) WHERE e.id = t.id AND origin.iso_code = 'BR' AND dest.iso_code = 'GB' RETURN ai.name AS ingredient, origin.name AS exporting_country, dest.name AS importing_country LIMIT 25",
    "explanation": "Finds active ingredients exported from Brazil to the UK using ISO code matching for both countries."
  }
"""

# ---------------------------------------------------------------------------
# NEWS section (included when NEWS label present)
# ---------------------------------------------------------------------------

_PROMPT_NEWS = """\
  NEWS domain context:
    NEWS nodes are articles/reports that connect to other entities via:
      NEWS -[MENTIONS_ACTIVE_INGREDIENT]-> ACTIVE_INGREDIENT
      NEWS -[MENTIONS_COUNTRY]->           COUNTRY
      NEWS -[MENTIONS_COMMODITY]->         COMMODITY
      NEWS -[MENTIONS_CURRENCY]->          CURRENCY
    Use NEWS as the starting node when the query asks about news, articles,
    mentions, or media coverage. Traverse from NEWS outward to find which
    entities a piece of news is about.

NEWS examples:

  Query   : "what news mentions Glyphosate?"
  Resolved entities: {"active_ingredients": {"Glyphosate": "Glyphosate"}}
  Output  : {
    "cypher": "MATCH (n:NEWS)-[:MENTIONS_ACTIVE_INGREDIENT]->(ai:ACTIVE_INGREDIENT) WHERE toLower(ai.name) = toLower('Glyphosate') RETURN n.title AS news_title, n.date AS date, ai.name AS ingredient ORDER BY n.date DESC LIMIT 25",
    "explanation": "Finds all news articles that mention Glyphosate via the MENTIONS_ACTIVE_INGREDIENT relationship, using exact name matching."
  }

  Query   : "which news articles mention both India and a fungicide?"
  Resolved entities: {"countries": {"India": "IN"}, "active_ingredients": {}}
  Output  : {
    "cypher": "MATCH (n:NEWS)-[:MENTIONS_COUNTRY]->(c:COUNTRY), (n)-[:MENTIONS_ACTIVE_INGREDIENT]->(ai:ACTIVE_INGREDIENT) WHERE c.iso_code = 'IN' AND toLower(ai.type) CONTAINS 'fungicide' RETURN n.title AS news_title, ai.name AS ingredient, c.name AS country LIMIT 25",
    "explanation": "Finds news that co-mentions India and a fungicide active ingredient using ISO code matching for the country."
  }
"""

# ---------------------------------------------------------------------------
# ALERT section (included when ALERT label present)
# ---------------------------------------------------------------------------

_PROMPT_ALERT = """\
  ALERT domain context:
    ALERT nodes are alerts generated by UPL's commodity alert system. Each alert
    is linked to the affected commodity via:
      ALERT -[AFFECTS_COMMODITY]-> COMMODITY
    Properties: severity (STRING), status (STRING), alert_type (STRING),
    title (STRING), id (STRING), created_at (STRING).
    Use ALERT as the starting node when the query asks about alerts, warnings,
    risk signals, or notifications. Traverse to COMMODITY to identify which
    commodity the alert concerns.

  ALERT QUERY RULE -- For ANY query involving alerts, warnings, risk signals,
  or notifications:
    a. Always start from ALERT and traverse to COMMODITY:
           (a:ALERT)-[:AFFECTS_COMMODITY]->(c:COMMODITY)
    b. Use toLower() for all string property filters:
           WHERE toLower(a.severity)   = 'critical'
           WHERE toLower(a.status)     = 'open'
           WHERE toLower(a.alert_type) CONTAINS 'price'
    c. Always RETURN a.title, a.severity, a.status, a.alert_type, a.created_at,
       and c.name so the full alert context and affected commodity are visible.
    d. Default LIMIT is 25 unless the user specifies otherwise.

ALERT examples:

  Query   : "show me all critical alerts"
  Output  : {
    "cypher": "MATCH (a:ALERT)-[:AFFECTS_COMMODITY]->(c:COMMODITY) WHERE toLower(a.severity) = 'critical' RETURN a.title AS alert_title, a.severity AS severity, a.status AS status, a.alert_type AS alert_type, a.created_at AS created_at, c.name AS commodity LIMIT 25",
    "explanation": "Finds all ALERT nodes with critical severity and returns their details along with the affected commodity."
  }

  Query   : "are there any open price alerts for wheat?"
  Output  : {
    "cypher": "MATCH (a:ALERT)-[:AFFECTS_COMMODITY]->(c:COMMODITY) WHERE toLower(a.status) = 'open' AND toLower(a.alert_type) CONTAINS 'price' AND toLower(c.name) CONTAINS 'wheat' RETURN a.title AS alert_title, a.severity AS severity, a.status AS status, a.alert_type AS alert_type, a.created_at AS created_at, c.name AS commodity LIMIT 25",
    "explanation": "Finds open price-type alerts affecting wheat by filtering on alert status, alert_type, and commodity name."
  }
"""


# ---------------------------------------------------------------------------
# Prompt builder -- assembles only the sections relevant to ranked labels
# ---------------------------------------------------------------------------

def build_system_prompt(label_names: list[str]) -> str:
    """Assemble the system prompt from conditional sections based on active labels."""
    sections = [_PROMPT_BASE]

    # Entity matching rules (always needed)
    sections.append(_PROMPT_MATCHING)

    # Trade corridors -- needed when both COUNTRY and ACTIVE_INGREDIENT are present
    if "COUNTRY" in label_names and "ACTIVE_INGREDIENT" in label_names:
        sections.append(_PROMPT_TRADE)

    # NEWS -- needed when NEWS label is present
    if "NEWS" in label_names:
        sections.append(_PROMPT_NEWS)

    # ALERT -- needed when ALERT label is present
    if "ALERT" in label_names:
        sections.append(_PROMPT_ALERT)

    return "\n".join(sections)

# ---------------------------------------------------------------------------
# Query planner -- decides single vs multi-query strategy
# ---------------------------------------------------------------------------

_PROMPT_PLANNER = """\
You are a query planner for a Neo4j knowledge graph. Your job is to decide
whether a user question can be answered with a single Cypher query, or whether
it should be decomposed into multiple independent sub-queries whose results
will be combined.

Strategies:

1. SINGLE -- Use when:
   - The question is about one traversal pattern (even if it has multiple filters)
   - The question naturally maps to one MATCH pattern with WHERE clauses
   - The sub-parts share the same starting node and direction

2. MULTI -- Decompose when:
   - The question asks about UNRELATED domains that don't share a traversal path
     (e.g. "show alerts for wheat AND which countries export Glyphosate")
   - The question asks for both an aggregation AND individual details
     (e.g. "how many ingredients does India import, and list the top 5?")
   - The question has clearly independent clauses joined by "and", "also", "plus"
     that would require Cartesian products or UNION in a single query.

3. EXPLORE -- Use for broad, open-ended questions about a specific entity:
   - "tell me about Glyphosate", "what do you know about India",
     "give me all info on Soybean", "summarize everything on Atrazine"
   - These ask for a COMPREHENSIVE overview across multiple graph domains
   - Decompose into 3-5 focused analytical sub-queries that cover:
     a. Trade/export data (if entity is an active ingredient or country)
     b. Import data (who receives/sends it)
     c. Brand associations (what products contain it)
     d. Alerts or news (if relevant labels exist)
     e. Commodity connections (if applicable)
   - Each sub-query should be phrased as an ANALYTICAL question
     (e.g. "top 10 countries exporting X" not "list countries exporting X")
     so that results are chart-friendly (rankings, counts, comparisons).

Return ONLY a JSON object:
  - If single:  {"strategy": "single"}
  - If multi:   {"strategy": "multi", "sub_queries": ["sub-question 1", ...]}
  - If explore: {"strategy": "explore", "sub_queries": ["sub-question 1", ...], "entity": "the entity being explored"}

Each sub_query should be a self-contained natural-language question that can
be independently converted to Cypher. Maximum 5 sub-queries for explore, 3 for multi.
"""


def plan_query(
    query: str,
    label_names: list[str],
    client: AzureOpenAI | None = None,
) -> dict:
    """
    Decide whether the query needs a single Cypher or multiple sub-queries.

    Returns {"strategy": "single"} or
            {"strategy": "multi", "sub_queries": [...]}
    """
    if client is None:
        client = AzureOpenAI(
            api_key=AZURE_KEY, api_version=AZURE_VERSION, azure_endpoint=AZURE_ENDPOINT,
        )

    user_msg = (
        f'User question: "{query}"\n'
        f'Available graph labels: {label_names}'
    )

    try:
        resp = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_PLANNER},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        strategy = data.get("strategy", "single")
        if strategy == "explore":
            sub_queries = data.get("sub_queries", [])
            entity = data.get("entity", "")
            if sub_queries and len(sub_queries) <= 5:
                logger.info("Stage 4 -- Planner: EXPLORE '%s' (%d sub-queries)", entity, len(sub_queries))
                return {"strategy": "explore", "sub_queries": sub_queries, "entity": entity}
        elif strategy == "multi":
            sub_queries = data.get("sub_queries", [])
            if sub_queries and len(sub_queries) <= 3:
                logger.info("Stage 4 -- Planner: MULTI (%d sub-queries)", len(sub_queries))
                return {"strategy": "multi", "sub_queries": sub_queries}

        logger.info("Stage 4 -- Planner: SINGLE")
        return {"strategy": "single"}

    except Exception as exc:
        logger.warning("Stage 4 -- Planner failed (%s), defaulting to single.", exc)
        return {"strategy": "single"}


# ---------------------------------------------------------------------------
# Zero-result fallback -- verify entity existence
# ---------------------------------------------------------------------------

_VALIDATION_CYPHER_TEMPLATE = (
    'MATCH (n) WHERE ANY(lbl IN labels(n) WHERE lbl IN {labels}) '
    'AND toLower(n.name) CONTAINS toLower($term) '
    'RETURN labels(n) AS labels, n.name AS name LIMIT 5'
)


def _check_entity_exists(driver, query: str, label_names: list[str]) -> str | None:
    """
    When a query returns 0 results, check if the key entities even exist.
    Returns a hint string if we find something useful, else None.
    """
    # Extract quoted terms or capitalized words as candidate entity names
    candidates = re.findall(r'"([^"]+)"', query)
    if not candidates:
        candidates = [w for w in query.split() if w[0:1].isupper() and len(w) > 2]

    if not candidates:
        return None

    hints = []
    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            for term in candidates[:3]:  # check at most 3 terms
                cypher = (
                    f"MATCH (n) WHERE ANY(lbl IN labels(n) WHERE lbl IN {label_names}) "
                    f"AND toLower(n.name) CONTAINS toLower('{term}') "
                    f"RETURN labels(n) AS labels, n.name AS name LIMIT 3"
                )
                result = list(s.run(cypher))
                if not result:
                    hints.append(f'"{term}" was not found in the graph.')
                else:
                    names = [r["name"] for r in result]
                    hints.append(f'"{term}" matched: {names}')
    except Exception as exc:
        logger.warning("Validation query failed: %s", exc)
        return None

    return "\n".join(hints) if hints else None


# ---------------------------------------------------------------------------
# Query Relaxation -- simplify overly complex queries that return 0 results
# ---------------------------------------------------------------------------

_RELAX_SYSTEM_PROMPT = """\
You are a Neo4j Cypher expert. A query was generated to answer the user's question
but it returned 0 rows. Your job is to SIMPLIFY the query so it returns results.

Strategies (apply one or more):
  1. Remove the most restrictive WHERE clause (especially secondary filters)
  2. Replace exact equality (=) with CONTAINS for string matching
  3. Remove relationship property filters (keep only node filters)
  4. Reduce JOIN depth -- if traversing 3+ hops, shorten to 2
  5. Widen the LIMIT
  6. Remove ORDER BY if it references a computed/aggregate field that may be empty

Constraints:
  - The relaxed query MUST still be a valid READ-ONLY Cypher query
  - It should still attempt to answer the user's original question
  - Keep at least the primary MATCH pattern and one meaningful WHERE filter
  - Never add CREATE, MERGE, SET, DELETE, DETACH, or REMOVE
  - Always include a LIMIT (max 50)

Return ONLY a JSON object:
{
  "relaxed_cypher": "the simplified Cypher query",
  "explanation": "what was removed/simplified and why",
  "confidence": "high" | "medium" | "low"  -- how likely this will return results
}

If the query cannot be meaningfully simplified (already minimal), return:
{"relaxed_cypher": "", "explanation": "Query is already minimal", "confidence": "low"}
"""


def relax_and_retry(
    original_cypher: str,
    query: str,
    driver,
    client: AzureOpenAI,
    validation_hint: str | None = None,
) -> dict:
    """
    Attempt to simplify a zero-result Cypher query and re-execute.

    Args:
        original_cypher  : the Cypher that returned 0 rows
        query            : the user's original question
        driver           : Neo4j driver
        client           : AzureOpenAI client
        validation_hint  : entity-existence hint (if available)

    Returns:
        {"cypher": str, "results": list[dict], "explanation": str, "relaxed": bool}
    """
    user_msg = f'User question: "{query}"\n\n'
    user_msg += f'Original Cypher (returned 0 rows):\n{original_cypher}\n'
    if validation_hint:
        user_msg += f'\nEntity existence check:\n{validation_hint}\n'

    try:
        resp = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": _RELAX_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        relaxed_cypher = data.get("relaxed_cypher", "").strip()
        explanation = data.get("explanation", "")

        if not relaxed_cypher:
            logger.info("Stage 4 -- Relaxation: query already minimal, cannot simplify.")
            return {"cypher": "", "results": [], "explanation": explanation, "relaxed": False}

        if not _is_safe_read(relaxed_cypher):
            logger.error("Stage 4 -- Relaxation: unsafe Cypher blocked.")
            return {"cypher": "", "results": [], "explanation": "Relaxed query was unsafe.", "relaxed": False}

        # Execute the relaxed query
        results = execute_cypher(driver, relaxed_cypher)
        logger.info(
            "Stage 4 -- Relaxation: %d row(s) from simplified query. Change: %s",
            len(results), explanation
        )

        return {
            "cypher": relaxed_cypher,
            "results": results,
            "explanation": explanation,
            "relaxed": True,
        }

    except (RateLimitError, APIConnectionError) as exc:
        logger.warning("Stage 4 -- Relaxation LLM call failed: %s", exc)
    except Exception as exc:
        logger.error("Stage 4 -- Relaxation error: %s", exc)

    return {"cypher": "", "results": [], "explanation": "Relaxation failed.", "relaxed": False}


# ---------------------------------------------------------------------------
# LLM -- generate Cypher
# ---------------------------------------------------------------------------

def generate_cypher(
    query: str,
    ranked_labels: list[dict],
    schema: str,
    node_properties: str,
    client: AzureOpenAI | None = None,
    resolved_entities: dict | None = None,
    conversation_history: list[dict] | None = None,
    rel_properties: str | None = None,
) -> dict:
    """
    Ask the LLM to generate a Cypher query for query using the ranked labels.

    Args:
        resolved_entities : optional dict from Stage 2, e.g.
                            {"countries": {"USA": "US", "India": "IN"}}
        rel_properties    : optional relationship property catalogue from
                            get_relationship_properties()

    Returns {"cypher": str, "explanation": str}.
    Returns {"cypher": "", "explanation": "..."} on failure.
    """
    if client is None:
        client = AzureOpenAI(
            api_key        = AZURE_KEY,
            api_version    = AZURE_VERSION,
            azure_endpoint = AZURE_ENDPOINT,
        )

    label_names = [e["name"] for e in ranked_labels]

    # Scope schema and properties to only the relevant labels
    scoped_schema = filter_schema(schema, label_names)
    scoped_props  = filter_node_properties(node_properties, label_names)

    # Format ranked labels for the prompt
    ranked_str = "\n".join(
        f"  {i+1}. {e['name']}" + (f" -- {e['reason']}" if e.get("reason") else "")
        for i, e in enumerate(ranked_labels)
    )

    # Build resolved-entities section
    country_map = (resolved_entities or {}).get("countries", {})
    if country_map:
        resolved_str = "Resolved entities (use iso_code for these countries):\n"
        resolved_str += "\n".join(
            f'  "{mention}" -> iso_code = "{iso}"'
            for mention, iso in country_map.items()
        )
    else:
        resolved_str = "Resolved entities: (none -- use name-based matching for countries)"

    # Build conversation history block (last 3 turns) for follow-up query context
    history_block = ""
    if conversation_history:
        turns = conversation_history[-3:]
        lines = [f"Previous conversation ({len(turns)} turn(s) -- use for context on follow-up queries):"]
        for t in turns:
            ans_preview = t["answer"][:300].replace("\n", " ")
            suffix = "..." if len(t["answer"]) > 300 else ""
            lines.append(f'  Q: {t["query"]}')
            lines.append(f"  A: {ans_preview}{suffix}")
        history_block = "\n".join(lines) + "\n\n"

    # Scope relationship properties to relevant labels
    rel_props_block = ""
    if rel_properties:
        scoped_rel_props = filter_relationship_properties(rel_properties, scoped_schema, label_names)
        if scoped_rel_props and scoped_rel_props != "No relationship properties found.":
            rel_props_block = f'Relationship properties:\n{scoped_rel_props}\n\n'

    user_msg = (
        f'{history_block}'
        f'User query       : "{query}"\n\n'
        f'Relevant labels (ranked):\n{ranked_str}\n\n'
        f'Schema           :\n{scoped_schema}\n\n'
        f'Node properties  :\n{scoped_props}\n\n'
        f'{rel_props_block}'
        f'{resolved_str}'
    )

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model           = AZURE_MODEL,
                messages        = [
                    {"role": "system", "content": build_system_prompt(label_names)},
                    {"role": "user",   "content": user_msg},
                ],
                temperature     = 0,
                response_format = {"type": "json_object"},
            )
            raw  = resp.choices[0].message.content or "{}"
            data = json.loads(raw)

            cypher      = data.get("cypher", "").strip()
            explanation = data.get("explanation", "")

            if not cypher:
                logger.warning("Stage 4 -- LLM returned empty Cypher.")
                return {"cypher": "", "explanation": "LLM returned no query."}

            if not _is_safe_read(cypher):
                logger.error("Stage 4 -- unsafe Cypher blocked: %s", cypher)
                return {"cypher": "", "explanation": "Generated query contained write operations and was blocked."}

            logger.info("Stage 4 -- Cypher      : %s", cypher)
            logger.info("Stage 4 -- Explanation : %s", explanation)
            return {"cypher": cypher, "explanation": explanation}

        except (RateLimitError, APIConnectionError) as exc:
            logger.warning("Stage 4 attempt %d failed: %s -- retrying in %ds.", attempt, exc, 2 * attempt)
            time.sleep(2 * attempt)

        except (APIError, Exception) as exc:
            logger.error("Stage 4 error: %s", exc)
            break

    return {"cypher": "", "explanation": "Cypher generation failed after retries."}


# ---------------------------------------------------------------------------
# Neo4j -- execute Cypher
# ---------------------------------------------------------------------------

def execute_cypher(driver, cypher: str) -> list[dict]:
    """
    Run a read-only Cypher query and return rows as plain dicts.
    Returns an empty list on error.
    """
    if not cypher:
        return []
    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            result = s.run(cypher)
            rows   = [dict(record) for record in result]
        logger.info("Stage 4 -- %d row(s) returned.", len(rows))
        return rows
    except Exception as exc:
        logger.error("Stage 4 -- Cypher execution error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API -- single call for Stage 4
# ---------------------------------------------------------------------------

def query_graph(
    query: str,
    ranked_labels: list[dict],
    schema: str,
    node_properties: str,
    driver,
    client: AzureOpenAI | None = None,
    resolved_entities: dict | None = None,
    conversation_history: list[dict] | None = None,
    rel_properties: str | None = None,
) -> dict:
    """
    Full Stage 4 pipeline:
      1. Plan: decide single vs multi-query strategy
      2. Generate Cypher (one or more sub-queries)
      3. Execute against Neo4j
      4. If zero results, run entity-existence validation
      5. Return combined result

    Returns:
        {
            "cypher"      : str (or list[str] for multi),
            "explanation" : str (or list[str] for multi),
            "results"     : list[dict],
            "validation"  : str | None  (hint when 0 results returned),
        }
    """
    if client is None:
        client = AzureOpenAI(
            api_key=AZURE_KEY, api_version=AZURE_VERSION, azure_endpoint=AZURE_ENDPOINT,
        )

    label_names = [e["name"] for e in ranked_labels]

    # --- Step 1: Plan ---
    plan = plan_query(query, label_names, client)

    # --- Step 2+3: Generate & Execute ---
    if plan["strategy"] in ("multi", "explore"):
        all_cypher = []
        all_explanations = []
        results_by_query = []  # list of lists -- one per sub-query

        for sub_q in plan["sub_queries"]:
            gen = generate_cypher(
                sub_q, ranked_labels, schema, node_properties,
                client, resolved_entities, conversation_history,
                rel_properties,
            )
            rows = execute_cypher(driver, gen["cypher"]) if gen["cypher"] else []
            all_cypher.append(gen["cypher"])
            all_explanations.append(gen["explanation"])
            results_by_query.append(rows)

        # Flat list for total row count
        all_results = [row for rows in results_by_query for row in rows]

        # --- Step 4: Zero-result fallback ---
        validation = None
        if not all_results:
            validation = _check_entity_exists(driver, query, label_names)

        return {
            "cypher"          : all_cypher,
            "explanation"     : all_explanations,
            "results"         : all_results,
            "results_by_query": results_by_query,
            "validation"      : validation,
            "strategy"        : plan["strategy"],
        }

    else:
        # Single-query path (original behavior)
        gen = generate_cypher(
            query, ranked_labels, schema, node_properties,
            client, resolved_entities, conversation_history,
            rel_properties,
        )
        results = execute_cypher(driver, gen["cypher"]) if gen["cypher"] else []

        # --- Step 4: Zero-result → relaxation retry ---
        validation = None
        relaxed_info = None
        if not results and gen["cypher"]:
            validation = _check_entity_exists(driver, query, label_names)
            # Attempt query relaxation
            relaxed_info = relax_and_retry(
                gen["cypher"], query, driver, client,
                validation_hint=validation,
            )
            if relaxed_info["results"]:
                logger.info("Stage 4 -- Relaxed query succeeded with %d rows.", len(relaxed_info["results"]))
                results = relaxed_info["results"]
                # Update cypher/explanation to reflect what actually worked
                gen["cypher"] = relaxed_info["cypher"]
                gen["explanation"] = (
                    f"{gen['explanation']} [Relaxed: {relaxed_info['explanation']}]"
                )
                validation = None  # Clear validation since we got results

        return {
            "cypher"     : gen["cypher"],
            "explanation": gen["explanation"],
            "results"    : results,
            "validation" : validation,
            "relaxed"    : bool(relaxed_info and relaxed_info.get("relaxed") and relaxed_info.get("results")),
        }