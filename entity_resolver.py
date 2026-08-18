"""
entity_resolver.py  —  Stage 2
----------------------------------
Extracts entity mentions from the user query and resolves them to canonical
identifiers stored in the graph.

  - Countries          → ISO 3166-1 alpha-2 codes (matched against c.iso_code in Neo4j)
  - Active ingredients → title-cased canonical names for precise matching
  - Commodities        → trading symbols (e.g. "BL1:COM") via Neo4j full-text fuzzy match

Input  : query (str)
Output : dict — {
    "countries":          {"mention": "iso_code", ...},
    "active_ingredients": {"mention": "canonical_name", ...},
    "commodities":        {"mention": "symbol_or_None", ...}
}

If no entities are found, returns empty dicts for all keys.
The pipeline is unaffected by failures here — falls back to empty dicts (fail open).
"""
from _future_ import annotations

import json
import logging
import time

from openai import AzureOpenAI, RateLimitError, APIConnectionError, APIError

from config import AZURE_KEY, AZURE_VERSION, AZURE_ENDPOINT, AZURE_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ENTITY_RESOLVER_PROMPT = """\
You are an entity extraction and resolution assistant for UPL's macroeconomic knowledge graph.

Your task:
  Given a user query, extract:
  1. All country mentions → resolve each to its ISO 3166-1 alpha-2 code.
  2. All active ingredient mentions → return the canonical chemical name in title-case.

  Return ONLY a JSON object with two keys:
    - "countries"          : object mapping each country mention to its 2-letter ISO code
    - "active_ingredients" : object mapping each ingredient mention (as it appears in the
                             query) to its title-cased canonical form

  Country rules:
    - Handle common aliases, abbreviations, and alternate names:
        "USA", "US", "United States", "America"       -> "US"
        "UK", "Britain", "England", "Great Britain"   -> "GB"
        "UAE", "Emirates", "United Arab Emirates"     -> "AE"
        "South Korea", "Korea"                        -> "KR"
        "Russia", "Russian Federation"                -> "RU"
        "Iran", "Persia"                              -> "IR"
        "Vietnam", "Viet Nam"                         -> "VN"
        etc. -- apply the same logic to any country.
    - If no country is mentioned, return {"countries": {}}.
    - If a mention is ambiguous (e.g. "Korea" without qualifier), resolve to the
      most likely interpretation (North Korea -> "KP", South Korea -> "KR").
    - Extract only countries, not cities, regions, or continents.
    - Do NOT include "XX" entries -- omit any mention you cannot confidently resolve.

  Active ingredient rules:
    - Extract chemical compound names used in crop protection (herbicides, fungicides,
      insecticides, nematicides, plant growth regulators, etc.).
    - Normalise to title-case (e.g. "GLYPHOSATE" -> "Glyphosate", "glyphosate" -> "Glyphosate").
    - Do NOT extract brand names, commodity names (crops), or country names here.
    - Do NOT extract generic category or type terms — these describe classes of active
      ingredients, not specific compounds. Treat the following (and similar) as
      non-extractable categories:
        'pesticides', 'herbicides', 'fungicides', 'insecticides', 'nematicides',
        'agrochemicals', 'chemicals', 'crop protection products', 'active ingredients'
      Only extract specific chemical compound names (e.g. 'Glyphosate', 'Mancozeb',
      'Chlorpyrifos', 'Imidacloprid').
    - If no active ingredients are mentioned, return {"active_ingredients": {}}.

Examples:
  Query  : "which active ingredients does the USA export to India?"
  Output : {"countries": {"USA": "US", "India": "IN"}, "active_ingredients": {}}

  Query  : "what does UK import from Brazil?"
  Output : {"countries": {"UK": "GB", "Brazil": "BR"}, "active_ingredients": {}}

  Query  : "which countries import Glyphosate?"
  Output : {"countries": {}, "active_ingredients": {"Glyphosate": "Glyphosate"}}

  Query  : "news about glyphosate imports in India?"
  Output : {"countries": {"India": "IN"}, "active_ingredients": {"glyphosate": "Glyphosate"}}

  Query  : "trade between Emirates and South Korea"
  Output : {"countries": {"Emirates": "AE", "South Korea": "KR"}, "active_ingredients": {}}

  Query  : "which brands are sold in Europe?"
  Output : {"countries": {}, "active_ingredients": {}}

  Query  : "list all active ingredients"
  Output : {"countries": {}, "active_ingredients": {}}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_entities(
    query: str,
    client: AzureOpenAI | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Extract country and active ingredient mentions from query.

    Returns:
        {
            "countries":          {"mention": "iso_code", ...},
            "active_ingredients": {"mention": "canonical_name", ...},
        }

    Failure fallback (fail open):
        {"countries": {}, "active_ingredients": {}}  -- pipeline continues unaffected.
    """
    if client is None:
        client = AzureOpenAI(
            api_key        = AZURE_KEY,
            api_version    = AZURE_VERSION,
            azure_endpoint = AZURE_ENDPOINT,
        )

    # Inject last turn so references like "that country" can be resolved
    context_prefix = ""
    if conversation_history:
        last = conversation_history[-1]
        context_prefix = f'Previous query: "{last["query"]}"\n\n'
    user_content = f'{context_prefix}Current query: "{query}"'

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model           = AZURE_MODEL,
                messages        = [
                    {"role": "system", "content": ENTITY_RESOLVER_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature     = 0,
                response_format = {"type": "json_object"},
                max_tokens      = 300,
            )
            raw  = resp.choices[0].message.content or "{}"
            data = json.loads(raw)

            # Sanitise: drop any "XX" fallbacks the LLM may have included
            countries = {
                str(mention): str(iso).upper()
                for mention, iso in data.get("countries", {}).items()
                if iso and str(iso).upper() not in ("XX", "")
            }
            active_ingredients = {
                str(mention): str(canonical)
                for mention, canonical in data.get("active_ingredients", {}).items()
                if canonical
            }

            logger.info("Stage 2 -- resolved countries: %s", countries)
            logger.info("Stage 2 -- resolved active ingredients: %s", active_ingredients)
            return {"countries": countries, "active_ingredients": active_ingredients}

        except (RateLimitError, APIConnectionError) as exc:
            logger.warning(
                "Stage 2 attempt %d failed: %s -- retrying in %ds.",
                attempt, exc, 2 * attempt,
            )
            time.sleep(2 * attempt)

        except (APIError, Exception) as exc:
            logger.error("Stage 2 error: %s", exc)
            break

    logger.warning("Stage 2 -- entity resolution failed, defaulting to empty.")
    return {"countries": {}, "active_ingredients": {}}