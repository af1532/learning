#!/usr/bin/env python3
"""
Market Research Agent - Food Ingredients & Plant Nutrition M&A Intelligence

Conducts ongoing M&A target research across:
  - Acidulants (citric acid, lactic acid, phosphoric acid, etc.)
  - Preservatives (natural & synthetic, clean-label alternatives)
  - Texturizers (hydrocolloids, modified starches, emulsifiers)
  - Plant Nutrition (biostimulants, plant extracts, bio-fertilizers)

Usage:
  python market_research_agent.py              # run full research session
  python market_research_agent.py --preview    # print report to stdout only

Schedule (cron example - weekly Monday 7am):
  0 7 * * 1 cd /path/to/project && /usr/bin/python3 market_research_agent.py >> logs/agent.log 2>&1

Requires:
  ANTHROPIC_API_KEY environment variable
  pip install anthropic>=0.40.0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8000
MAX_TURNS = 30           # safety ceiling for the agentic loop
MAX_SEARCH_USES = 12     # hard cap on web searches per run
REPORTS_DIR = Path("reports")

# Beta header required for web_search_20250305
BETA_HEADER = "web-search-2025-03-05"

SYSTEM_PROMPT = """
You are a senior M&A research analyst specialising in the food ingredients and plant
nutrition industry. Your expertise covers:

1. ACIDULANTS — citric acid, lactic acid, acetic acid, tartaric acid, malic acid,
   phosphoric acid, fumaric acid, GDL, and emerging bio-based acidulants.

2. PRESERVATIVES — sodium benzoate, potassium sorbate, propionates, nitrites/nitrates,
   natural preservatives (rosemary extract, nisin, natamycin), and clean-label systems.

3. TEXTURIZERS — hydrocolloids (xanthan, carrageenan, pectin, guar gum, locust bean gum),
   native & modified starches, emulsifiers, stabilisers, and plant-based texture systems.

4. PLANT NUTRITION — biostimulants, plant amino acids, humic/fulvic acids,
   bio-fertilisers, and ingredients at the food/feed/agriculture intersection.

Your mandate:
• Identify companies that are attractive M&A targets in these four segments.
• Track recent deal activity, valuations, and buyer appetite.
• Surface consolidation trends and strategic rationales.
• Deliver concise, actionable intelligence for a corporate development team.

Always use current information (last 6–12 months). Be specific: name companies,
quote deal values, cite growth signals, flag PE-backed or founder-led businesses
likely to seek an exit. Structure every report with clear headers and tables.
"""

RESEARCH_PROMPT = """
Conduct a comprehensive M&A target intelligence report for the food ingredients and
plant nutrition market. Use web search extensively to surface the most current data.

---

## Section 1 — Recent M&A Deal Activity (Last 12 Months)

Search for and list completed acquisitions, mergers, minority investments, and
strategic partnerships in:
  - Food acidulants, preservatives, texturizers
  - Plant nutrition / biostimulants

For each deal provide:
| Date | Buyer | Target | Deal Value | Segment | Strategic Rationale |
|------|-------|--------|-----------|---------|---------------------|

---

## Section 2 — Identified M&A Targets

Research companies ($10 M – $500 M revenue range) that show acquisition potential:
  - Strong IP or proprietary formulations
  - High growth trajectory or niche market leadership
  - PE-backed (approaching fund lifecycle end) or founder-led
  - Geographic expansion stories (Asia, LatAm, MEA manufacturing)
  - Clean-label / natural / sustainable positioning

For each target provide:
| Company | Segment | HQ | Est. Revenue | Key Differentiator | Exit Signal | Likelihood |
|---------|---------|----|----|---|---|---|

---

## Section 3 — Active Strategic Buyers

Who is actively acquiring in this space right now?
  - Large ingredient companies making bolt-on acquisitions
  - Private equity firms with food ingredient theses
  - Their stated acquisition criteria and recent deal track record

---

## Section 4 — Market Trends Driving Consolidation

Explain the macro and regulatory forces creating M&A pressure:
  - Clean-label / natural demand
  - Sustainability & bio-based ingredients
  - Supply-chain regionalisation
  - Regulatory changes (EU, US, Asia)
  - Price volatility in commodity raw materials

---

## Section 5 — Valuation Benchmarks

Current trading and transaction multiples for food ingredient businesses:
  - EV/EBITDA ranges by sub-segment
  - EV/Revenue ranges
  - Notable premium/discount drivers

---

Conclude with a short EXECUTIVE SUMMARY (5–7 bullet points) of the top insights
and the 3 highest-conviction M&A targets identified.
"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def extract_text(content: list) -> str:
    """Pull all TextBlock text from a message content list."""
    return "\n".join(
        block.text
        for block in content
        if hasattr(block, "text") and block.text
    )


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Core research session
# ---------------------------------------------------------------------------

def run_research_session(verbose: bool = True) -> str:
    """
    Run one full M&A research session using Claude + web_search_20250305.

    web_search_20250305 is a SERVER-SIDE tool — Anthropic's infrastructure
    executes every search automatically. The stop_reason contract differs from
    client-side custom tools:

      "pause_turn"  — Server paused after executing one or more web searches.
                      Search results are already embedded in response.content as
                      web_search_tool_result blocks. Append the assistant turn
                      and re-call the API. Do NOT send tool_result messages.

      "end_turn"    — Claude is finished. Extract text and return.

      "tool_use"    — Defensive: handles any client-side tool_use that slips
                      through (should not occur for web_search in normal use).

      "max_tokens"  — Token ceiling hit mid-response; return partial result.
    """
    client = anthropic.Anthropic()

    log("=" * 60, verbose)
    log("MARKET RESEARCH AGENT  |  Food Ingredients M&A", verbose)
    log(f"Model : {MODEL}", verbose)
    log("=" * 60, verbose)

    messages: list[dict] = [{"role": "user", "content": RESEARCH_PROMPT}]

    for turn in range(1, MAX_TURNS + 1):
        log(f"Turn {turn:02d} — calling API ...", verbose)

        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": MAX_SEARCH_USES,
                }
            ],
            messages=messages,
            betas=[BETA_HEADER],
        )

        stop = response.stop_reason
        log(f"Turn {turn:02d} — stop_reason={stop}", verbose)

        # Log any web searches that fired this turn
        if verbose:
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "tool_use" and getattr(block, "name", "") == "web_search":
                    query = ""
                    if hasattr(block, "input") and isinstance(block.input, dict):
                        query = block.input.get("query", "")
                    log(f'  → web_search("{query[:80]}")', verbose)

        # Add assistant turn to history
        messages.append({"role": "assistant", "content": response.content})

        # ── Server paused after executing searches — just re-call ─────────
        if stop == "pause_turn":
            # Search results are already in response.content (injected by the
            # server). Do NOT add a tool_result user message — that would cause
            # a 400 error. Simply re-call the API so Claude can continue.
            continue

        # ── Done ──────────────────────────────────────────────────────────
        if stop == "end_turn":
            result = extract_text(response.content)
            log(f"Research complete after {turn} turn(s).", verbose)
            return result

        # ── Token ceiling ─────────────────────────────────────────────────
        if stop == "max_tokens":
            log("Warning: max_tokens reached — returning partial result.", verbose)
            return extract_text(response.content) or "[INCOMPLETE: max_tokens reached]"

        # ── Defensive: client-side tool_use (should not fire for web_search) ─
        if stop == "tool_use":
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "[Tool not available client-side]",
                            "is_error": True,
                        }
                    )
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            continue

        # ── Unknown stop reason ───────────────────────────────────────────
        log(f"Unexpected stop_reason='{stop}' — stopping loop.", verbose)
        break

    # Fallback: return whatever text is in the last assistant message
    log("Warning: reached max turns — returning partial result.", verbose)
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            text = extract_text(content if isinstance(content, list) else [])
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def build_report(content: str) -> str:
    now = datetime.utcnow()
    header = (
        "# Food Ingredients & Plant Nutrition — M&A Intelligence Report\n\n"
        f"**Generated:** {now.strftime('%B %d, %Y  %H:%M UTC')}  \n"
        "**Segments:** Acidulants · Preservatives · Texturizers · Plant Nutrition  \n"
        "**Focus:** M&A Target Identification  \n\n"
        "---\n\n"
    )
    return header + content


def save_report(content: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = REPORTS_DIR / f"market_research_{ts}.md"
    path.write_text(build_report(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Food Ingredients M&A Research Agent")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print report to stdout instead of saving to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    research = run_research_session(verbose=verbose)

    if not research.strip():
        print("[ERROR] No content generated. Check your ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    report = build_report(research)

    if args.preview:
        print(report)
    else:
        path = save_report(research)
        print(f"\nReport saved → {path}")

        # Print executive summary preview (first 40 lines)
        lines = report.splitlines()
        preview = "\n".join(lines[:40])
        print("\n--- PREVIEW (first 40 lines) ---")
        print(preview)
        if len(lines) > 40:
            print(f"... ({len(lines) - 40} more lines in full report)")


if __name__ == "__main__":
    main()
