"""
add_iso_codes.py
----------------
Fetches all Country nodes from Neo4j, uses Azure OpenAI to determine
the correct ISO 3166-1 alpha-2 code for each country, and writes
the iso_code property back to each node.

Usage:
    python add_iso_codes.py
"""
from _future_ import annotations

import json
import logging
import time

from neo4j import GraphDatabase
from openai import AzureOpenAI

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    AZURE_KEY, AZURE_VERSION, AZURE_ENDPOINT, AZURE_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def get_llm_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_KEY,
        api_version=AZURE_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


# ---------------------------------------------------------------------------
# Fetch all Country nodes
# ---------------------------------------------------------------------------

def fetch_country_names(driver) -> list[dict]:
    """
    Returns a list of dicts: [{"id": <neo4j_id>, "name": <country_name>}, ...]
    """
    cypher = """
        MATCH (c:Country)
        RETURN elementId(c) AS id, c.name AS name
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        results = list(session.run(cypher))
    countries = [{"id": r["id"], "name": r["name"]} for r in results]
    logger.info(f"Found {len(countries)} Country nodes.")
    return countries


# ---------------------------------------------------------------------------
# Use LLM to resolve ISO codes (batched)
# ---------------------------------------------------------------------------

def resolve_iso_codes(client: AzureOpenAI, country_names: list[str]) -> dict[str, str]:
    """
    Sends a batch of country names to the LLM and returns a mapping:
    {country_name: iso_alpha2_code}
    """
    prompt = (
        "You are given a list of country names. For each one, return the "
        "ISO 3166-1 alpha-2 code. Respond ONLY with valid JSON — a single "
        "object mapping each country name (exactly as given) to its 2-letter "
        "ISO code. If a name is ambiguous or not a real country, use \"XX\".\n\n"
        f"Country names:\n{json.dumps(country_names)}"
    )

    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": "You are a geography expert. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith(""):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("", 1)[0]

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Write iso_code property back to Neo4j
# ---------------------------------------------------------------------------

def update_iso_codes(driver, mappings: list[dict]):
    """
    mappings: [{"id": elementId, "iso_code": "US"}, ...]
    """
    cypher = """
        UNWIND $rows AS row
        MATCH (c:Country) WHERE elementId(c) = row.id
        SET c.iso_code = row.iso_code
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(cypher, rows=mappings)
    logger.info(f"Updated {len(mappings)} Country nodes with iso_code.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    driver = get_driver()
    client = get_llm_client()

    countries = fetch_country_names(driver)
    if not countries:
        logger.warning("No Country nodes found. Exiting.")
        return

    # Batch into groups of 40 to stay within token limits
    BATCH_SIZE = 40
    all_mappings: list[dict] = []

    for i in range(0, len(countries), BATCH_SIZE):
        batch = countries[i : i + BATCH_SIZE]
        names = [c["name"] for c in batch]

        logger.info(f"Resolving ISO codes for batch {i // BATCH_SIZE + 1} "
                    f"({len(names)} countries)...")

        iso_map = resolve_iso_codes(client, names)

        for c in batch:
            code = iso_map.get(c["name"], "XX")
            all_mappings.append({"id": c["id"], "iso_code": code})

        # Small delay between batches to respect rate limits
        if i + BATCH_SIZE < len(countries):
            time.sleep(1)

    # Write all codes back to Neo4j
    update_iso_codes(driver, all_mappings)

    # Print summary
    logger.info("Done! Sample results:")
    for m in all_mappings[:10]:
        matching = next((c for c in countries if c["id"] == m["id"]), None)
        if matching:
            print(f"  {matching['name']:<30} -> {m['iso_code']}")

    driver.close()


if __name__ == "_main_":
    main()