"""
visualizer.py  —  Stage 7 (Enhanced)
--------------------------------------
LLM-driven semantic visualization for Graph RAG query results.

Instead of blindly plotting based on column types, this module:
  1. Classifies query INTENT (analytical vs lookup) to decide IF a chart is needed
  2. Uses an LLM chart planner to decide WHAT to plot and WHY
  3. Applies domain-aware axis labels (ISO codes → country names, column keys → business terms)
  4. Handles multi-query results intelligently (charts each sub-query independently)
  5. Falls back to heuristic detection when LLM is unavailable

Input  : results, query, semantic context (explanation, resolved_entities, ranked_labels)
Output : dict — {"visualized": bool, "charts": list[dict], "reason": str}
"""
from _future_ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import (
    VIZ_MIN_ROWS, VIZ_MAX_PIE_CATS, VIZ_OUTPUT_DIR, VIZ_OPEN_BROWSER,
    VIZ_LLM_ENABLED, AZURE_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(VIZ_OUTPUT_DIR)
MIN_ROWS_FOR_VIZ = VIZ_MIN_ROWS
MAX_PIE_CATEGORIES = VIZ_MAX_PIE_CATS

# ---------------------------------------------------------------------------
# Domain-aware label mapping
# ---------------------------------------------------------------------------

COLUMN_LABEL_MAP = {
    "iso_code": "Country (ISO)", "country_name": "Country", "country": "Country",
    "c.name": "Country", "c.iso_code": "Country (ISO)",
    "exporter": "Exporting Country", "importer": "Importing Country",
    "ai_name": "Active Ingredient", "ai.name": "Active Ingredient",
    "ingredient": "Active Ingredient", "active_ingredient": "Active Ingredient",
    "brand_name": "Brand", "b.name": "Brand",
    "commodity_name": "Commodity", "commodity": "Commodity", "com.name": "Commodity",
    "count": "Count", "total": "Total", "volume": "Volume",
    "export_volume": "Export Volume", "import_volume": "Import Volume",
    "num_ingredients": "Number of Ingredients", "num_countries": "Number of Countries",
    "severity": "Severity", "alert_type": "Alert Type", "status": "Status",
    "created_at": "Date Created", "title": "Title", "source": "Source",
    "date": "Date", "published_date": "Published Date",
    "currency_code": "Currency", "cur.code": "Currency", "symbol": "Trading Symbol",
}

ISO_TO_COUNTRY = {
    "US": "United States", "IN": "India", "CN": "China", "BR": "Brazil",
    "AR": "Argentina", "AU": "Australia", "DE": "Germany", "FR": "France",
    "GB": "United Kingdom", "JP": "Japan", "MX": "Mexico", "ZA": "South Africa",
    "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam", "MY": "Malaysia",
    "CO": "Colombia", "CL": "Chile", "PE": "Peru", "PH": "Philippines",
    "KR": "South Korea", "IT": "Italy", "ES": "Spain", "NL": "Netherlands",
    "BE": "Belgium", "PL": "Poland", "TR": "Turkey", "EG": "Egypt",
    "NG": "Nigeria", "KE": "Kenya", "PK": "Pakistan", "BD": "Bangladesh",
    "UA": "Ukraine", "RU": "Russia", "CA": "Canada", "NZ": "New Zealand",
    "AE": "UAE", "SA": "Saudi Arabia", "IL": "Israel", "TW": "Taiwan",
}


# ---------------------------------------------------------------------------
# 1. Intent Classification
# ---------------------------------------------------------------------------

_ANALYTICAL_SIGNALS = [
    r"\bhow many\b", r"\bcount\b", r"\btop \d+", r"\bbottom \d+",
    r"\bcompare\b", r"\bcomparison\b", r"\bdistribution\b",
    r"\btrend\b", r"\bover time\b", r"\bgrowth\b", r"\bdecline\b",
    r"\brank\b", r"\branking\b", r"\bmost\b", r"\bleast\b",
    r"\bhighest\b", r"\blowest\b", r"\btotal\b", r"\bsum\b",
    r"\baverage\b", r"\bpercentage\b", r"\bproportion\b",
    r"\bbreakdown\b", r"\bby country\b", r"\bby region\b",
    r"\bper\b", r"\beach\b",
]

_LOOKUP_SIGNALS = [
    r"\blist\b", r"\bshow me\b", r"\bwhat is\b", r"\bwhat are\b",
    r"\bwhich\b.*\bcontain", r"\btell me about\b", r"\bdescribe\b",
    r"\bdetails\b", r"\binformation\b", r"\bname\b.*\bof\b",
    r"\bexplain\b", r"\bdefine\b",
]


def classify_intent(query: str, results: list[dict]) -> str:
    """Classify query intent as 'analytical', 'lookup', or 'ambiguous'."""
    q_lower = query.lower()
    analytical_score = sum(1 for p in _ANALYTICAL_SIGNALS if re.search(p, q_lower))
    lookup_score = sum(1 for p in _LOOKUP_SIGNALS if re.search(p, q_lower))

    if results:
        col_types = _classify_columns(results)
        numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
        if not numeric_cols and len(results) <= 5:
            return "lookup"

    if analytical_score > lookup_score:
        return "analytical"
    elif lookup_score > analytical_score:
        return "lookup"
    else:
        if results:
            col_types = _classify_columns(results)
            numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
            if numeric_cols:
                return "analytical"
        return "lookup"


# ---------------------------------------------------------------------------
# 2. LLM Chart Planner
# ---------------------------------------------------------------------------

_VIZ_PLANNER_SYSTEM = """You are a data visualization expert for UPL's agrochemical knowledge graph.

Given a user query, its Cypher explanation, column names with sample values, and resolved entities,
decide whether and how to visualize the results.

Rules:
- Only recommend a chart if it adds analytical value (comparisons, rankings, distributions, trends)
- Do NOT chart simple lookups or enumerations
- Choose the chart type that best communicates the insight
- Provide human-readable axis labels (not raw column keys)
- If country ISO codes are used, expand them to full names in the label

Chart types: line, bar, pie, count_bar, scatter

Respond with JSON only:
{
  "should_visualize": true/false,
  "chart_type": "bar" | "line" | "pie" | "count_bar" | "scatter" | null,
  "x": "column_name" | null,
  "y": "column_name" | null,
  "color": "column_name" | null,
  "title": "Descriptive chart title",
  "x_label": "Human-readable x-axis label",
  "y_label": "Human-readable y-axis label",
  "reasoning": "Brief explanation"
}"""


def _build_planner_prompt(query, results, explanation=None, resolved_entities=None, ranked_labels=None):
    columns = list(results[0].keys()) if results else []
    sample_rows = results[:3]
    parts = [f"User query: {query}"]
    if explanation:
        if isinstance(explanation, list):
            parts.append(f"Cypher explanations: {'; '.join(str(e) for e in explanation)}")
        else:
            parts.append(f"Cypher explanation: {explanation}")
    if resolved_entities:
        countries = resolved_entities.get("countries", {})
        ingredients = resolved_entities.get("active_ingredients", {})
        if countries:
            parts.append(f"Resolved countries: {countries}")
        if ingredients:
            parts.append(f"Resolved ingredients: {ingredients}")
    if ranked_labels:
        label_names = [l.get("name", "") for l in ranked_labels[:3]]
        parts.append(f"Primary graph labels: {', '.join(label_names)}")
    parts.append(f"\nResult columns: {columns}")
    parts.append(f"Total rows: {len(results)}")
    parts.append(f"Sample data (first 3 rows):")
    for i, row in enumerate(sample_rows, 1):
        parts.append(f"  Row {i}: {row}")
    return "\n".join(parts)


def llm_plan_chart(query, results, client, explanation=None, resolved_entities=None, ranked_labels=None):
    """Ask the LLM to plan the visualization. Returns chart config or None."""
    if not client or not results:
        return None

    user_prompt = _build_planner_prompt(query, results, explanation, resolved_entities, ranked_labels)

    for attempt in range(1, 4):
        try:
            from openai import RateLimitError, APIConnectionError
            resp = client.chat.completions.create(
                model=AZURE_MODEL,
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _VIZ_PLANNER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            plan = json.loads(raw)

            if not plan.get("should_visualize"):
                logger.info("Stage 7 — LLM planner: skip. Reason: %s", plan.get("reasoning", ""))
                return {"should_visualize": False, "reasoning": plan.get("reasoning", "")}

            if plan.get("x") and plan["x"] not in results[0]:
                logger.warning("Stage 7 — LLM suggested x='%s' not in columns.", plan["x"])
                return None
            if plan.get("y") and plan["y"] not in results[0] and plan.get("chart_type") != "count_bar":
                logger.warning("Stage 7 — LLM suggested y='%s' not in columns.", plan["y"])
                return None

            logger.info("Stage 7 — LLM planner: %s (%s)", plan.get("chart_type"), plan.get("reasoning", ""))
            return plan

        except (RateLimitError, APIConnectionError):
            time.sleep(2 * attempt)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Stage 7 — LLM planner parse error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Stage 7 — LLM planner failed: %s", exc)
            return None

    logger.warning("Stage 7 — LLM planner exhausted retries.")
    return None


# ---------------------------------------------------------------------------
# 3. Column Type Classification (heuristic fallback)
# ---------------------------------------------------------------------------

def _is_numeric(value) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except (ValueError, AttributeError):
            return False
    return False


def _is_date(value) -> bool:
    if not isinstance(value, str):
        return False
    date_patterns = [r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}", r"\d{4}-\d{2}-\d{2}T"]
    return any(re.match(p, value) for p in date_patterns)


def _classify_columns(results: list[dict]) -> dict:
    if not results:
        return {}
    sample = results[:10]
    columns = list(results[0].keys())
    classification = {}
    for col in columns:
        values = [row.get(col) for row in sample if row.get(col) is not None]
        if not values:
            classification[col] = "categorical"
            continue
        numeric_count = sum(1 for v in values if _is_numeric(v))
        date_count = sum(1 for v in values if _is_date(v))
        if date_count > len(values) * 0.7:
            classification[col] = "date"
        elif numeric_count > len(values) * 0.7:
            classification[col] = "numeric"
        else:
            classification[col] = "categorical"
    return classification


def heuristic_detect(results: list[dict], query: str = "") -> dict | None:
    """Heuristic fallback for chart type selection."""
    if not results or len(results) < MIN_ROWS_FOR_VIZ:
        return None

    col_types = _classify_columns(results)
    columns = list(col_types.keys())
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]

    if date_cols and numeric_cols:
        return {
            "should_visualize": True, "chart_type": "line",
            "x": date_cols[0], "y": numeric_cols[0],
            "color": cat_cols[0] if cat_cols else None,
            "title": _generate_title(query, "line"),
            "x_label": _get_label(date_cols[0]), "y_label": _get_label(numeric_cols[0]),
            "reasoning": "Time-series pattern detected",
        }

    if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        unique_cats = len(set(row.get(cat_cols[0], "") for row in results))
        if unique_cats <= MAX_PIE_CATEGORIES and len(columns) == 2:
            return {
                "should_visualize": True, "chart_type": "pie",
                "x": cat_cols[0], "y": numeric_cols[0], "color": None,
                "title": _generate_title(query, "pie"),
                "x_label": _get_label(cat_cols[0]), "y_label": _get_label(numeric_cols[0]),
                "reasoning": f"Distribution with {unique_cats} categories",
            }
        return {
            "should_visualize": True, "chart_type": "bar",
            "x": cat_cols[0], "y": numeric_cols[0],
            "color": cat_cols[1] if len(cat_cols) > 1 else None,
            "title": _generate_title(query, "bar"),
            "x_label": _get_label(cat_cols[0]), "y_label": _get_label(numeric_cols[0]),
            "reasoning": "Categorical comparison with numeric values",
        }

    if cat_cols and not numeric_cols and len(results) >= MIN_ROWS_FOR_VIZ:
        return {
            "should_visualize": True, "chart_type": "count_bar",
            "x": cat_cols[0], "y": None,
            "color": cat_cols[1] if len(cat_cols) > 1 else None,
            "title": _generate_title(query, "bar"),
            "x_label": _get_label(cat_cols[0]), "y_label": "Count",
            "reasoning": "Frequency distribution of categorical values",
        }

    return None


# ---------------------------------------------------------------------------
# 4. Smart Axis Labels & ISO Expansion
# ---------------------------------------------------------------------------

def _get_label(col_name: str) -> str:
    if col_name in COLUMN_LABEL_MAP:
        return COLUMN_LABEL_MAP[col_name]
    return col_name.replace("_", " ").replace(".", " ").strip().title()


def _expand_iso_codes_in_data(df, col_name, resolved_entities=None):
    """If a column has ISO codes, add a display column with full country names."""
    import pandas as pd
    sample_values = df[col_name].dropna().head(10).tolist()
    if not sample_values:
        return df, col_name
    looks_like_iso = all(isinstance(v, str) and len(v) == 2 and v.isupper() for v in sample_values)
    if not looks_like_iso:
        return df, col_name

    lookup = dict(ISO_TO_COUNTRY)
    if resolved_entities:
        for mention, iso in resolved_entities.get("countries", {}).items():
            if iso not in lookup:
                lookup[iso] = mention.title()

    display_col = f"{col_name}_name"
    df[display_col] = df[col_name].map(lambda v: lookup.get(v, v))
    return df, display_col


def _generate_title(query: str, chart_type: str) -> str:
    title = query.strip().rstrip("?").strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title.capitalize() if title else f"Query Results ({chart_type} chart)"


# ---------------------------------------------------------------------------
# 5. Multi-Query Handler
# ---------------------------------------------------------------------------

def _select_best_subquery(results_by_query: list[list[dict]]) -> list[tuple[int, list[dict]]]:
    """Return sub-queries worth charting."""
    chartable = []
    for i, rows in enumerate(results_by_query):
        if not rows or len(rows) < MIN_ROWS_FOR_VIZ:
            continue
        col_types = _classify_columns(rows)
        numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
        date_cols = [c for c, t in col_types.items() if t == "date"]
        cat_cols = [c for c, t in col_types.items() if t == "categorical"]
        if numeric_cols or date_cols or (cat_cols and len(rows) >= MIN_ROWS_FOR_VIZ):
            chartable.append((i, rows))
    return chartable


# ---------------------------------------------------------------------------
# 6. Chart Generation
# ---------------------------------------------------------------------------

def generate_chart(results, config, resolved_entities=None, open_browser=True):
    """Generate and save a Plotly chart. Returns path or None."""
    if not PLOTLY_AVAILABLE:
        return None

    try:
        import pandas as pd
        df = pd.DataFrame(results)
    except ImportError:
        logger.error("Stage 7 — pandas not available.")
        return None

    chart_type = config["chart_type"]
    x_col = config["x"]
    y_col = config.get("y")
    color_col = config.get("color")
    title = config.get("title", "Query Results")
    x_label = config.get("x_label", _get_label(x_col) if x_col else "")
    y_label = config.get("y_label", _get_label(y_col) if y_col else "")

    try:
        fig = None
        display_x = x_col
        if x_col:
            df, display_x = _expand_iso_codes_in_data(df, x_col, resolved_entities)

        if chart_type == "bar":
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            fig = px.bar(df, x=display_x, y=y_col, color=color_col, title=title,
                         labels={display_x: x_label, y_col: y_label})
            fig.update_layout(xaxis_tickangle=-45)

        elif chart_type == "count_bar":
            counts = df[display_x].value_counts().reset_index()
            counts.columns = [display_x, "count"]
            fig = px.bar(counts, x=display_x, y="count", title=title,
                         labels={display_x: x_label, "count": y_label or "Count"})
            fig.update_layout(xaxis_tickangle=-45)

        elif chart_type == "pie":
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            fig = px.pie(df, names=display_x, values=y_col, title=title)

        elif chart_type == "line":
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            df = df.sort_values(x_col)
            fig = px.line(df, x=display_x, y=y_col, color=color_col, title=title,
                          labels={display_x: x_label, y_col: y_label}, markers=True)

        elif chart_type == "scatter":
            df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title,
                             labels={x_col: x_label, y_col: y_label})

        if fig is None:
            return None

        fig.update_layout(
            template="plotly_white", font=dict(size=12),
            title_font=dict(size=16), margin=dict(l=50, r=30, t=60, b=80),
        )

        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{chart_type}_{timestamp}.html"
        filepath = OUTPUT_DIR / filename
        fig.write_html(str(filepath))
        logger.info("Stage 7 — chart saved: %s", filepath)

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{filepath.resolve()}")

        return str(filepath)

    except Exception as exc:
        logger.error("Stage 7 — chart generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API — single call for Stage 7
# ---------------------------------------------------------------------------

def visualize_results(
    results: list[dict],
    query: str,
    client=None,
    explanation=None,
    resolved_entities=None,
    ranked_labels=None,
    results_by_query=None,
    strategy: str | None = None,
    open_browser: bool = True,
) -> dict:
    """
    Full Stage 7 pipeline (enhanced):
      1. Classify query intent (analytical vs lookup)
      2. LLM planner or heuristic fallback
      3. Handle multi-query results independently
      4. Generate chart(s) with smart labels and ISO expansion

    Returns:
        {"visualized": bool, "charts": list[dict], "reason": str}
    """
    if not PLOTLY_AVAILABLE:
        return {"visualized": False, "charts": [], "reason": "plotly not installed"}

    if not results:
        return {"visualized": False, "charts": [], "reason": "No results to visualize."}

    # Step 1: Intent classification (bypass for explore strategy)
    if strategy != "explore":
        intent = classify_intent(query, results)
        if intent == "lookup":
            logger.info("Stage 7 — intent='lookup', skipping.")
            return {"visualized": False, "charts": [], "reason": "Query is a lookup/enumeration — chart not appropriate."}
    else:
        logger.info("Stage 7 — explore strategy: bypassing intent check, forcing chart generation.")

    # Step 2: Handle multi-query results
    chart_tasks = []
    if results_by_query and len(results_by_query) > 1:
        chartable = _select_best_subquery(results_by_query)
        if not chartable:
            return {"visualized": False, "charts": [], "reason": "No sub-query has a chartable shape."}
        for idx, rows in chartable:
            chart_tasks.append((idx, rows))
    else:
        flat = results
        if flat and isinstance(flat[0], list):
            flat = [row for sublist in flat for row in sublist]
        if len(flat) < MIN_ROWS_FOR_VIZ:
            return {"visualized": False, "charts": [], "reason": f"Too few rows ({len(flat)})."}
        chart_tasks.append((None, flat))

    # Step 3: Plan + generate charts
    generated_charts = []
    for sub_idx, rows in chart_tasks:
        config = None

        # Try LLM planner
        if VIZ_LLM_ENABLED and client:
            sub_explanation = explanation
            if isinstance(explanation, list) and sub_idx is not None:
                sub_explanation = explanation[sub_idx] if sub_idx < len(explanation) else None
            config = llm_plan_chart(query, rows, client, explanation=sub_explanation,
                                    resolved_entities=resolved_entities, ranked_labels=ranked_labels)

        if config and config.get("should_visualize") is False:
            continue

        # Heuristic fallback
        if config is None:
            config = heuristic_detect(rows, query)

        if config is None or not config.get("should_visualize", True):
            continue

        path = generate_chart(rows, config, resolved_entities=resolved_entities, open_browser=open_browser)
        if path:
            generated_charts.append({
                "chart_type": config["chart_type"],
                "path": path,
                "reasoning": config.get("reasoning", ""),
                "sub_query_index": sub_idx,
                "title": config.get("title", ""),
            })

    if generated_charts:
        return {"visualized": True, "charts": generated_charts, "reason": f"{len(generated_charts)} chart(s) generated"}
    else:
        return {"visualized": False, "charts": [], "reason": _explain_skip(results)}


def _explain_skip(results: list[dict]) -> str:
    if not results:
        return "No results to visualize."
    if len(results) < MIN_ROWS_FOR_VIZ:
        return f"Too few rows ({len(results)}) — need at least {MIN_ROWS_FOR_VIZ}."
    col_types = _classify_columns(results)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    if not numeric_cols:
        unique_vals = len(set(str(row.get(list(col_types.keys())[0], "")) for row in results))
        if unique_vals < MIN_ROWS_FOR_VIZ:
            return "Not enough distinct values for a meaningful chart."
    return "Data shape not suitable for standard chart types."