"""
main.py  --  Graph RAG Orchestrator
------------------------------------
Runs the full pipeline:

  Stage 1    relevance_gate.py   -- Guard: is the query answerable from the graph?
  Stage 2    entity_resolver.py  -- Extract + resolve country mentions to ISO codes
  Stage 3    label_filter.py     -- Filter + rank relevant Neo4j labels
  Stage 4    cypher_gen.py       -- Generate + execute a Cypher query (Text2Cypher)
  Stage 5    answer_gen.py       -- Format raw Neo4j rows into a Markdown table
  Stage 6    answer_gen.py       -- Generate a natural-language answer via LLM
  Stage 7    visualizer.py       -- (Optional) Generate a chart from the results

Usage:
    python main.py "which countries import Glyphosate?"
    python main.py           # prompts interactively
"""
from _future_ import annotations

import logging
import sys

from openai import AzureOpenAI
from neo4j.exceptions import ServiceUnavailable, AuthError
from rich.console import Console
from rich.markdown import Markdown

from config import (
    AZURE_KEY, AZURE_VERSION, AZURE_ENDPOINT,
    MAX_HISTORY, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    VIZ_ENABLED, VIZ_OPEN_BROWSER,
)

from neo4j_helpers    import get_driver, get_labels, get_schema, get_node_properties, get_relationship_properties
from relevance_gate   import check_relevance
from entity_resolver  import resolve_entities
from label_filter     import get_relevant_labels
from cypher_gen       import query_graph
from answer_gen       import generate_answer
from visualizer       import visualize_results

MAX_HISTORY = 5  # max conversation turns to retain in context

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    stream = sys.stderr,
)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logger  = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_badge(confidence: float) -> str:
    """Return a Rich-coloured confidence badge string."""
    pct = int(confidence * 100)
    if confidence >= CONFIDENCE_HIGH:
        return f"[bold green]{pct}%[/bold green]"
    if confidence >= CONFIDENCE_MEDIUM:
        return f"[bold yellow]{pct}%[/bold yellow]"
    return f"[bold red]{pct}%[/bold red]"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(query: str, history: list[dict]) -> list[dict]:
    # Shared LLM client -- one instance reused across all stages
    client = AzureOpenAI(
        api_key        = AZURE_KEY,
        api_version    = AZURE_VERSION,
        azure_endpoint = AZURE_ENDPOINT,
    )

    # -- Stage 1 -- Relevance gate -------------------------------------------
    console.rule("[bold blue]Stage 1 -- Relevance Check")
    gate       = check_relevance(query, client, conversation_history=history)
    relevant   = gate["relevant"]
    confidence = gate["confidence"]
    reason     = gate["reason"]
    badge      = _confidence_badge(confidence)

    if not relevant:
        # High-confidence refusal -- firm message
        if confidence >= CONFIDENCE_HIGH:
            console.print(
                f"\n[bold yellow]Outside scope[/bold yellow] "
                f"(confidence {badge})\n"
                f"[dim]{reason}[/dim]"
            )
        # Low-confidence refusal -- softer, suggest rephrasing
        else:
            console.print(
                f"\n[bold yellow]Likely outside scope[/bold yellow] "
                f"(confidence {badge}) -- the query is ambiguous.\n"
                f"[dim]{reason}[/dim]\n"
                f"[dim]Try rephrasing with specific agrochemical terms "
                f"(e.g. active ingredient, brand, commodity, country).[/dim]"
            )
        console.print(
            "\n[dim]The graph covers active ingredients, brands, commodities, "
            "countries, and currencies in the agrochemical supply chain.[/dim]\n"
        )
        return history

    # Relevant -- show badge and warn if borderline
    console.print(f"[green]Relevant[/green] (confidence {badge}) -- {reason}")
    if confidence < CONFIDENCE_MEDIUM:
        console.print(
            "[yellow]Warning:[/yellow] [dim]low confidence -- the graph may only "
            "partially cover this query.[/dim]"
        )
    console.print()

    # -- Stage 2 -- Entity resolution ----------------------------------------
    console.rule("[bold blue]Stage 2 -- Entity Resolution")
    resolved    = resolve_entities(query, client, conversation_history=history)
    country_map = resolved.get("countries", {})

    if country_map:
        console.print("[bold cyan]Resolved countries:[/bold cyan]")
        for mention, iso in country_map.items():
            console.print(f"  [green]{mention}[/green] -> [yellow]{iso}[/yellow]")
    else:
        console.print("[dim]No country mentions detected -- name-based matching will be used.[/dim]")

    ingredient_map = resolved.get("active_ingredients", {})
    if ingredient_map:
        console.print("[bold cyan]Resolved active ingredients:[/bold cyan]")
        for mention, canonical in ingredient_map.items():
            console.print(f"  [green]{mention}[/green] -> [yellow]{canonical}[/yellow]")
    console.print()

    # Connect to Neo4j (only after the gate passes)
    try:
        driver = get_driver()
    except (ServiceUnavailable, AuthError) as exc:
        console.print(f"[bold red]Neo4j connection failed:[/bold red] {exc}")
        sys.exit(1)

    try:
        # -- Fetch graph metadata (shared across stages 3-4) -----------------
        all_labels      = get_labels(driver)
        schema          = get_schema(driver)
        node_properties = get_node_properties(driver)
        rel_properties  = get_relationship_properties(driver)

        # -- Stage 3 -- Label filtering & ranking ----------------------------
        console.rule("[bold blue]Stage 3 -- Label Filtering")
        ranked = get_relevant_labels(query, all_labels, schema, node_properties, client)

        console.print("\n[bold cyan]Relevant labels (ranked):[/bold cyan]")
        for i, entry in enumerate(ranked, start=1):
            name   = entry.get("name", "")
            reason = entry.get("reason", "")
            line   = f"  [bold]{i}.[/bold] [green]{name:<25}[/green]"
            if reason:
                line += f" -- {reason}"
            console.print(line)

        # -- Stage 4 -- Cypher generation & execution ------------------------
        console.rule("[bold blue]Stage 4 -- Cypher Generation")
        stage2 = query_graph(
            query, ranked, schema, node_properties, driver, client,
            resolved_entities=resolved,
            conversation_history=history,
            rel_properties=rel_properties,
        )

        cypher = stage2["cypher"]
        console.print(f"\n[bold cyan]Generated Cypher:[/bold cyan]")
        if isinstance(cypher, list):
            for i, c in enumerate(cypher, 1):
                console.print(f"  [yellow]Sub-query {i}:[/yellow] {c}")
            console.print(f"\n[bold cyan]Explanations:[/bold cyan]")
            for i, e in enumerate(stage2["explanation"], 1):
                console.print(f"  {i}. {e}")
        else:
            console.print(f"  [yellow]{cypher}[/yellow]")
            console.print(f"\n[bold cyan]Explanation:[/bold cyan] {stage2['explanation']}")
        console.print(f"\n[bold cyan]Rows returned:[/bold cyan] {len(stage2['results'])}")

        # Show validation hint if present (zero-result fallback)
        if stage2.get("validation"):
            console.print(f"\n[bold yellow]Validation:[/bold yellow] {stage2['validation']}")

        if isinstance(cypher, list) and not any(cypher):
            console.print("[red]Cypher generation failed -- cannot proceed to answer.[/red]")
            return history
        elif isinstance(cypher, str) and not cypher:
            console.print("[red]Cypher generation failed -- cannot proceed to answer.[/red]")
            return history

        # -- Stage 5 + 6 -- Answer generation --------------------------------
        console.rule("[bold blue]Stage 5+6 -- Answer Generation")
        answer = generate_answer(
            query, stage2["cypher"], stage2["results"], client,
            conversation_history=history,
            validation=stage2.get("validation"),
            results_by_query=stage2.get("results_by_query"),
        )

        console.print()
        console.print(Markdown(answer))
        console.print()

        # -- Stage 7 -- Visualization (optional, toggle via VIZ_ENABLED in config)
        if VIZ_ENABLED and stage2["results"]:
            console.rule("[bold blue]Stage 7 -- Visualization")
            viz = visualize_results(
                results=stage2["results"],
                query=query,
                client=client,
                explanation=stage2.get("explanation"),
                resolved_entities=resolved,
                ranked_labels=ranked,
                results_by_query=stage2.get("results_by_query"),
                strategy=stage2.get("strategy"),
                open_browser=VIZ_OPEN_BROWSER,
            )
            if viz["visualized"]:
                for chart_info in viz["charts"]:
                    sub_label = f" (sub-query {chart_info['sub_query_index'] + 1})" if chart_info.get("sub_query_index") is not None else ""
                    console.print(
                        f"[bold green]\U0001f4ca Chart generated{sub_label}:[/bold green] "
                        f"{chart_info['chart_type']} chart \u2192 [link=file://{chart_info['path']}]{chart_info['path']}[/link]"
                    )
                    if chart_info.get("reasoning"):
                        console.print(f"  [dim]Reason: {chart_info['reasoning']}[/dim]")
            else:
                console.print(f"[dim]Stage 7 \u2014 no chart: {viz['reason']}[/dim]")

        history = (history + [{"query": query, "answer": answer}])[-MAX_HISTORY:]

    finally:
        driver.close()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    console.rule("[bold blue]Graph RAG Agent -- UPL Knowledge Graph")

    if args:
        # Single-shot mode: query passed as CLI argument (original behaviour)
        query = " ".join(args)
        if not query:
            console.print("[red]Error: query cannot be empty.[/red]")
            sys.exit(1)
        run(query, [])
    else:
        # Interactive loop mode: keep session alive between queries
        console.print("[dim]Type 'exit' or 'quit' to end the session.[/dim]\n")
        history: list[dict] = []
        while True:
            try:
                query = input("Enter your query: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[dim]Session ended.[/dim]")
                break
            history = run(query, history)


if __name__ == "_main_":
    main()