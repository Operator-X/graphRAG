"""
answer_gen.py  —  Stage 5 + Stage 6
-------------------------------------
Stage 5 : Format raw Neo4j result rows into a readable Markdown table.
Stage 6 : Pass the query + formatted results to the LLM and generate a
          concise natural-language answer.

Input  : query (str), cypher (str | list[str]), results (list[dict]),
         validation (str | None)
Output : str  — the final answer
"""
from _future_ import annotations

import logging

from openai import AzureOpenAI, APIError

from config import AZURE_KEY, AZURE_VERSION, AZURE_ENDPOINT, AZURE_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are a data analyst for UPL — an agrochemical company.

You have been given the results of a Neo4j graph query that was run to answer a
user's question about UPL's macroeconomic knowledge graph (active ingredients,
brands, commodities, countries, currencies).

Your task:
  Write a clear, concise answer to the user's question based strictly on the
  provided query results. Do not invent or assume any facts not present in the data.

Rules:
  1. If the results table is empty, say so clearly. If a "Validation hints"
     section is provided, use it to explain why (e.g. an entity was not found
     in the graph, or the relationship may not be modelled).
  2. Use Markdown formatting where it aids readability (bullet lists, bold text).
     Do not repeat the raw results table — summarise and highlight key findings.
  3. If the data contains numeric values, include relevant totals or comparisons.
  4. Keep the answer focused — avoid padding or restating the question.
"""

# ---------------------------------------------------------------------------
# Stage 5 — format results
# ---------------------------------------------------------------------------

def format_results(results: list[dict]) -> str:
    """
    Serialize Neo4j result rows as a Markdown table.
    Returns a plain "No results." message if the list is empty.
    """
    if not results:
        return "No results returned from the graph."

    headers = list(results[0].keys())

    # Header row
    header_row = "| " + " | ".join(headers) + " |"
    sep_row    = "| " + " | ".join("---" for _ in headers) + " |"

    data_rows = [
        "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
        for row in results
    ]

    return "\n".join([header_row, sep_row] + data_rows)


# ---------------------------------------------------------------------------
# Stage 6 — generate answer
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    cypher: str | list[str],
    results: list[dict],
    client: AzureOpenAI | None = None,
    conversation_history: list[dict] | None = None,
    validation: str | None = None,
    results_by_query: list[list[dict]] | None = None,
) -> str:
    """
    Generate a natural-language answer to query given the Neo4j results.

    Args:
        query            : original user question
        cypher           : the Cypher query(ies) executed (str or list[str] for multi)
        results          : raw rows from Neo4j (flat merged list)
        client           : optional pre-built AzureOpenAI client
        validation       : optional hint from zero-result fallback (entity existence check)
        results_by_query : for multi-query, results grouped per sub-query

    Returns a Markdown-formatted answer string.
    """
    if client is None:
        client = AzureOpenAI(
            api_key        = AZURE_KEY,
            api_version    = AZURE_VERSION,
            azure_endpoint = AZURE_ENDPOINT,
        )

    # Format results -- group by sub-query for multi-query, or flat for single
    if results_by_query and isinstance(cypher, list):
        results_sections = []
        for i, (sub_cypher, sub_rows) in enumerate(zip(cypher, results_by_query), 1):
            section = f"--- Sub-query {i} results ({len(sub_rows)} rows) ---\n"
            section += format_results(sub_rows)
            results_sections.append(section)
        results_text = "\n\n".join(results_sections)
    else:
        results_text = format_results(results)

    # Build conversation history block (last 3 turns) for coherent follow-up answers
    history_block = ""
    if conversation_history:
        turns = conversation_history[-3:]
        lines = [f"Previous conversation ({len(turns)} turn(s)):"]
        for t in turns:
            ans_preview = t["answer"][:300].replace("\n", " ")
            suffix = "..." if len(t["answer"]) > 300 else ""
            lines.append(f'  Q: {t["query"]}')
            lines.append(f"  A: {ans_preview}{suffix}")
        history_block = "\n".join(lines) + "\n\n"

    # Format cypher — handle both single string and multi-query list
    if isinstance(cypher, list):
        cypher_text = "\n".join(
            f"  Sub-query {i+1}: {c}" for i, c in enumerate(cypher) if c
        )
    else:
        cypher_text = cypher

    # Build validation hint block (only when results are empty)
    validation_block = ""
    if validation and not results:
        validation_block = f"\n\nValidation hints (entity existence check):\n{validation}"

    user_msg = (
        f"{history_block}"
        f"User question :\n{query}\n\n"
        f"Cypher query executed :\n{cypher_text}\n\n"
        f"Query results :\n{results_text}"
        f"{validation_block}"
    )

    try:
        resp = client.chat.completions.create(
            model       = AZURE_MODEL,
            messages    = [
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.3,
        )
        answer = resp.choices[0].message.content or ""
        logger.info("Stage 6 — answer generated (%d chars).", len(answer))
        return answer

    except (APIError, Exception) as exc:
        logger.error("Stage 6 — answer generation failed: %s", exc)
        # Graceful degradation: return the raw table so the user still gets something
        return (
            f"Answer generation failed ({exc}). Raw results below:\n\n"
            + results_text
        )