#!/usr/bin/env python3
"""
SEO Keyword Agent

Generates 200-300 SEO keywords for tracking a brand's search presence.
Accepts brand name, website domain, and optional seed keywords as inputs.

Usage:
    python keyword_agent.py "Notion" https://notion.so
    python keyword_agent.py "Figma" https://figma.com --seeds "design tool" "wireframing"
    python keyword_agent.py "Linear" https://linear.app --count 300 --output linear.csv

Search Volume:
    Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD env vars for real Google search volumes
    (sign up at https://dataforseo.com — free trial available).
    Without credentials, volume tiers are estimated by Claude.
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
from typing import Optional

import requests
from bs4 import BeautifulSoup
import anthropic

client = anthropic.Anthropic()


# ─── Web Fetching Tool ───────────────────────────────────────────────────────

def fetch_webpage(url: str, max_chars: int = 6000) -> str:
    """Fetch and extract readable text content from a webpage."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "noscript", "iframe", "aside", "form"]):
            tag.decompose()

        # Extract title and meta description for extra signal
        title = soup.find("title")
        meta_desc = soup.find("meta", attrs={"name": "description"})
        prefix = ""
        if title:
            prefix += f"PAGE TITLE: {title.get_text(strip=True)}\n"
        if meta_desc and meta_desc.get("content"):
            prefix += f"META DESCRIPTION: {meta_desc['content']}\n\n"

        # Extract h1/h2 headings for structure
        headings = []
        for h in soup.find_all(["h1", "h2"], limit=12):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)
        if headings:
            prefix += "HEADINGS: " + " | ".join(headings) + "\n\n"

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        return (prefix + text)[:max_chars]

    except requests.exceptions.HTTPError as e:
        return f"HTTP error fetching {url}: {e}"
    except requests.exceptions.ConnectionError:
        return f"Could not connect to {url} — site may be unreachable."
    except requests.exceptions.Timeout:
        return f"Timeout fetching {url}."
    except Exception as e:
        return f"Error fetching {url}: {e}"


TOOLS = [
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch and extract text content from a webpage. "
            "Use this to research a brand's website — fetch the homepage, "
            "product/features pages, about page, pricing page, blog, etc. "
            "Returns page title, headings, and body text to help understand "
            "what the brand does, their products/services, and target audience."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (must include https://)"
                }
            },
            "required": ["url"]
        }
    }
]


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "fetch_webpage":
        return fetch_webpage(tool_input["url"])
    return f"Unknown tool: {name}"


# ─── Phase 1: Brand Research ─────────────────────────────────────────────────

def research_brand(brand: str, domain: str) -> str:
    """
    Agentic research loop: fetches brand website pages and returns a
    comprehensive brand research summary.
    """
    print(f"\n[1/3] Researching {brand} at {domain}...", file=sys.stderr)

    messages = [
        {
            "role": "user",
            "content": (
                f"Research this brand thoroughly by fetching their website. "
                f"Gather detailed information about:\n"
                f"- What they do (products/services)\n"
                f"- Target audience/customers\n"
                f"- Key features, capabilities, and benefits\n"
                f"- Positioning and differentiators\n"
                f"- Use cases and verticals they serve\n"
                f"- Pricing model/tiers (if visible)\n"
                f"- Industry/category they operate in\n"
                f"- Key terminology and language they use\n\n"
                f"Brand: {brand}\n"
                f"Website: {domain}\n\n"
                f"Fetch the homepage and 3–5 key pages (about, products/features, "
                f"pricing, solutions, use cases, etc.). Then write a comprehensive "
                f"BRAND RESEARCH REPORT covering all findings above."
            )
        }
    ]

    while True:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return next(
                (b.text for b in response.content if b.type == "text"),
                "No brand research available."
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool in tool_use_blocks:
            url = tool.input.get("url", "")
            print(f"  → Fetching {url}", file=sys.stderr)
            result = execute_tool(tool.name, tool.input)
            print(f"    {len(result)} chars extracted", file=sys.stderr)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


# ─── Phase 2: Keyword Generation ─────────────────────────────────────────────

# Structured output schema for keyword list
KEYWORD_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "brand_summary": {
            "type": "string",
            "description": "One-sentence description of what the brand does"
        },
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The SEO keyword phrase"
                    },
                    "funnel_stage": {
                        "type": "string",
                        "enum": ["upper", "mid", "lower"],
                        "description": "Marketing funnel stage"
                    },
                    "category": {
                        "type": "string",
                        "description": "Thematic category (e.g., Product Features, Pricing, Alternatives)"
                    },
                    "intent": {
                        "type": "string",
                        "enum": ["informational", "commercial", "transactional", "navigational"],
                        "description": "Search intent"
                    }
                },
                "required": ["keyword", "funnel_stage", "category", "intent"],
                "additionalProperties": False
            }
        }
    },
    "required": ["brand_summary", "keywords"],
    "additionalProperties": False
}


def generate_keywords(
    brand: str,
    domain: str,
    brand_research: str,
    seed_keywords: Optional[list[str]],
    target_count: int,
) -> dict:
    """
    Generate SEO keywords using structured output. Returns raw dict with
    brand_summary and keywords list.
    """
    print(f"\n[2/3] Generating {target_count} keywords...", file=sys.stderr)

    seed_section = ""
    if seed_keywords:
        seed_section = (
            f"\n\nSeed keywords to incorporate and build upon:\n"
            + "\n".join(f"  - {k}" for k in seed_keywords)
        )

    # Target funnel distribution
    upper = round(target_count * 0.30)
    mid = round(target_count * 0.40)
    lower = target_count - upper - mid

    prompt = (
        f"Based on the brand research below, generate exactly {target_count} SEO "
        f"keywords for tracking {brand}'s search presence ({domain}).\n\n"
        f"BRAND RESEARCH:\n{brand_research}"
        f"{seed_section}\n\n"
        f"TARGET DISTRIBUTION:\n"
        f"  - Upper funnel (awareness): ~{upper} keywords\n"
        f"  - Mid funnel (consideration): ~{mid} keywords\n"
        f"  - Lower funnel (purchase): ~{lower} keywords\n\n"
        f"KEYWORD REQUIREMENTS:\n\n"
        f"Upper funnel — informational, problem-aware, educational:\n"
        f"  e.g., 'how to [solve problem brand solves]', 'what is [category]',\n"
        f"  '[industry] best practices', '[pain point] solutions'\n\n"
        f"Mid funnel — consideration, comparison, research:\n"
        f"  e.g., '[brand] features', 'best [category] software',\n"
        f"  '[brand] vs [competitor]', '[brand] review', '[feature] tool',\n"
        f"  '[use case] platform', '[audience] [category] software'\n\n"
        f"Lower funnel — purchase intent, brand + transactional:\n"
        f"  e.g., '[brand] pricing', '[brand] plans', '[brand] free trial',\n"
        f"  'buy [product]', '[brand] [feature] demo', '[brand] for [team size]'\n\n"
        f"CATEGORY GUIDELINES — assign each keyword a specific thematic category:\n"
        f"  Examples: 'Product Features', 'Pricing & Plans', 'Alternatives',\n"
        f"  'Use Cases', 'Integrations', 'Industry Terms', 'How-To Guides',\n"
        f"  'Brand', 'Comparisons', 'Problem Awareness', 'Team/Role-Specific'\n\n"
        f"QUALITY RULES:\n"
        f"  - Use real phrasing people type into Google (natural language)\n"
        f"  - Mix short-tail (1-2 words) and long-tail (3-5 words) keywords\n"
        f"  - Avoid overly broad single-word terms with no commercial signal\n"
        f"  - Avoid fabricated product names not found in the research\n"
        f"  - Include brand name variants and branded combinations\n"
        f"  - Incorporate industry jargon and audience-specific terminology\n\n"
        f"Generate exactly {target_count} keywords total."
    )

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={
            "format": {
                "type": "json_schema",
                "schema": KEYWORD_LIST_SCHEMA,
            }
        },
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    text = next((b.text for b in response.content if b.type == "text"), "")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from text
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(
                f"Could not parse keyword response as JSON.\n"
                f"Response (first 500 chars): {text[:500]}"
            )
        data = json.loads(match.group())

    count = len(data.get("keywords", []))
    print(f"  Generated {count} keywords", file=sys.stderr)
    return data


# ─── Phase 3: Volume Enrichment ──────────────────────────────────────────────

def get_dataforseo_volumes(
    keywords: list[str], location_code: int = 2840
) -> dict[str, Optional[int]]:
    """
    Fetch monthly search volumes from DataForSEO Keywords Data API.
    location_code 2840 = United States
    """
    login = os.environ.get("DATAFORSEO_LOGIN", "")
    password = os.environ.get("DATAFORSEO_PASSWORD", "")
    credentials = base64.b64encode(f"{login}:{password}".encode()).decode()

    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"
    volumes: dict[str, Optional[int]] = {}

    # DataForSEO accepts up to 1000 keywords per request
    for i in range(0, len(keywords), 1000):
        batch = keywords[i : i + 1000]
        payload = [{"keywords": batch, "location_code": location_code, "language_code": "en"}]

        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("tasks"):
                for task in data["tasks"]:
                    if task.get("result"):
                        for item in task["result"]:
                            kw = item.get("keyword", "")
                            vol = item.get("search_volume")
                            volumes[kw] = vol if vol is not None else 0

        except requests.exceptions.HTTPError as e:
            # 401 = bad credentials, 400 = bad request
            status = e.response.status_code if e.response else "unknown"
            print(
                f"  DataForSEO error (HTTP {status}): {e}. "
                f"Check DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD.",
                file=sys.stderr,
            )
            return {}
        except Exception as e:
            print(f"  DataForSEO error: {e}", file=sys.stderr)
            return {}

    return volumes


# Structured schema for Claude volume tier estimation
VOLUME_TIER_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "volume_tier": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low", "Very Low"],
                    },
                },
                "required": ["keyword", "volume_tier"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def estimate_volume_tiers(keywords_data: list[dict]) -> dict[str, str]:
    """
    Use Claude to estimate search volume tiers for keywords when DataForSEO
    credentials are not available.
    """
    kw_list = [
        {
            "keyword": kw["keyword"],
            "funnel_stage": kw["funnel_stage"],
            "intent": kw["intent"],
        }
        for kw in keywords_data
    ]

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": VOLUME_TIER_SCHEMA,
            }
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "Estimate the monthly Google search volume tier for each keyword below.\n\n"
                    "Volume tiers:\n"
                    "  High:     10,000+ monthly searches — generic category terms, "
                    "short popular phrases\n"
                    "  Medium:   1,000–9,999 monthly searches — specific feature "
                    "queries, mid-length phrases\n"
                    "  Low:      100–999 monthly searches — specific long-tail, "
                    "niche use-case terms\n"
                    "  Very Low: <100 monthly searches — hyper-specific long-tail, "
                    "obscure terms\n\n"
                    "Consider: keyword length, specificity, brand vs generic, "
                    "question format, commercial intent.\n\n"
                    f"Keywords:\n{json.dumps(kw_list, indent=2)}"
                ),
            }
        ],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            data = json.loads(match.group())
        else:
            return {}

    return {item["keyword"]: item["volume_tier"] for item in data.get("items", [])}


def _volume_tier(vol: int) -> str:
    if vol >= 10000:
        return "High"
    if vol >= 1000:
        return "Medium"
    if vol >= 100:
        return "Low"
    return "Very Low"


def enrich_with_volumes(keywords_data: list[dict]) -> list[dict]:
    """
    Add search_volume (int or None) and volume_tier (str) to each keyword dict.
    Uses DataForSEO if credentials are set, otherwise estimates with Claude.
    """
    has_dataforseo = bool(
        os.environ.get("DATAFORSEO_LOGIN") and os.environ.get("DATAFORSEO_PASSWORD")
    )

    if has_dataforseo:
        print("  Fetching volumes from DataForSEO...", file=sys.stderr)
        keyword_strs = [kw["keyword"] for kw in keywords_data]
        volumes = get_dataforseo_volumes(keyword_strs)

        if volumes:
            for kw in keywords_data:
                vol = volumes.get(kw["keyword"])
                kw["search_volume"] = vol
                kw["volume_tier"] = _volume_tier(vol) if vol is not None else ""
        else:
            # DataForSEO failed — fall back to estimates
            print(
                "  DataForSEO returned no data; falling back to Claude estimates...",
                file=sys.stderr,
            )
            _apply_claude_estimates(keywords_data)
    else:
        print(
            "  No DataForSEO credentials found — estimating tiers with Claude.\n"
            "  (Set DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD for real volume data.)",
            file=sys.stderr,
        )
        _apply_claude_estimates(keywords_data)

    return keywords_data


def _apply_claude_estimates(keywords_data: list[dict]):
    tiers = estimate_volume_tiers(keywords_data)
    for kw in keywords_data:
        kw["search_volume"] = None
        kw["volume_tier"] = tiers.get(kw["keyword"], "")


# ─── Output ──────────────────────────────────────────────────────────────────

def save_csv(keywords_data: list[dict], output_path: str):
    fieldnames = ["keyword", "funnel_stage", "category", "intent",
                  "search_volume", "volume_tier"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for kw in keywords_data:
            row = {k: kw.get(k, "") for k in fieldnames}
            if row["search_volume"] is None:
                row["search_volume"] = ""
            writer.writerow(row)


def print_report(brand_summary: str, keywords_data: list[dict], has_real_volumes: bool):
    n = len(keywords_data)
    width = 65

    print(f"\n{'='*width}")
    print("SEO KEYWORD REPORT")
    print(f"{'='*width}")
    print(f"Brand:          {brand_summary}")
    print(f"Total Keywords: {n}")

    # Funnel distribution
    funnel: dict[str, int] = {}
    for kw in keywords_data:
        s = kw.get("funnel_stage", "unknown")
        funnel[s] = funnel.get(s, 0) + 1

    print(f"\nFunnel Distribution:")
    for stage in ["upper", "mid", "lower"]:
        count = funnel.get(stage, 0)
        pct = count / n * 100 if n else 0
        bar = "█" * (count // max(n // 40, 1))
        print(f"  {stage.capitalize():<8} {count:>3}  ({pct:4.0f}%)  {bar}")

    # Top categories
    cats: dict[str, int] = {}
    for kw in keywords_data:
        c = kw.get("category", "Other")
        cats[c] = cats.get(c, 0) + 1

    print(f"\nTop Categories:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:12]:
        print(f"  {count:>3}  {cat}")

    # Intent breakdown
    intents: dict[str, int] = {}
    for kw in keywords_data:
        i = kw.get("intent", "unknown")
        intents[i] = intents.get(i, 0) + 1

    print(f"\nIntent Breakdown:")
    for intent in ["informational", "commercial", "transactional", "navigational"]:
        count = intents.get(intent, 0)
        if count:
            print(f"  {intent.capitalize():<18} {count:>3}")

    # Volume tiers
    tiers: dict[str, int] = {}
    for kw in keywords_data:
        t = kw.get("volume_tier", "")
        if t:
            tiers[t] = tiers.get(t, 0) + 1

    if tiers:
        label = "Search Volume (from DataForSEO)" if has_real_volumes else "Volume Tiers (estimated)"
        print(f"\n{label}:")
        for tier in ["High", "Medium", "Low", "Very Low"]:
            count = tiers.get(tier, 0)
            if count:
                pct = count / n * 100
                print(f"  {tier:<10} {count:>3}  ({pct:.0f}%)")

    # Sample keywords per funnel stage
    print(f"\nSample Keywords:")
    for stage in ["upper", "mid", "lower"]:
        stage_kws = [kw for kw in keywords_data if kw.get("funnel_stage") == stage][:5]
        if stage_kws:
            print(f"\n  {stage.upper()} FUNNEL:")
            for kw in stage_kws:
                vol_part = ""
                if kw.get("search_volume") is not None:
                    vol_part = f"  ~{kw['search_volume']:,}/mo"
                elif kw.get("volume_tier"):
                    vol_part = f"  [{kw['volume_tier']}]"
                print(f"    • {kw['keyword']}{vol_part}")

    print(f"\n{'='*width}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 200-300 SEO keywords for a brand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python keyword_agent.py \"Notion\" https://notion.so\n"
            "  python keyword_agent.py \"Figma\" https://figma.com "
            "--seeds \"design tool\" \"prototyping\"\n"
            "  python keyword_agent.py \"Linear\" https://linear.app "
            "--count 300 --output linear.csv\n\n"
            "Search Volume:\n"
            "  Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD for real Google "
            "search volumes.\n"
            "  Without these, volume tiers are estimated by Claude.\n"
            "  Sign up at https://dataforseo.com (free trial available)."
        ),
    )
    parser.add_argument("brand", help='Brand name (e.g., "Notion")')
    parser.add_argument("domain", help="Website URL (e.g., https://notion.so)")
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        metavar="KEYWORD",
        help="Optional seed keywords to build upon",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=250,
        help="Target keyword count, 200-300 recommended (default: 250)",
    )
    parser.add_argument(
        "--output", "-o",
        default="keywords.csv",
        help="Output CSV file path (default: keywords.csv)",
    )
    parser.add_argument(
        "--no-volumes",
        action="store_true",
        help="Skip search volume enrichment entirely",
    )

    args = parser.parse_args()

    # Normalize domain
    domain = args.domain
    if not domain.startswith("http"):
        domain = "https://" + domain

    if args.count < 50 or args.count > 1000:
        parser.error("--count must be between 50 and 1000")

    # Phase 1: Research
    brand_research = research_brand(args.brand, domain)

    # Phase 2: Generate keywords
    result = generate_keywords(
        brand=args.brand,
        domain=domain,
        brand_research=brand_research,
        seed_keywords=args.seeds,
        target_count=args.count,
    )

    keywords_data: list[dict] = result.get("keywords", [])
    brand_summary: str = result.get("brand_summary", args.brand)

    if not keywords_data:
        print("Error: No keywords were generated.", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Volume enrichment
    has_real_volumes = False
    if not args.no_volumes:
        print(f"\n[3/3] Enriching with search volume data...", file=sys.stderr)
        enrich_with_volumes(keywords_data)
        has_real_volumes = bool(
            os.environ.get("DATAFORSEO_LOGIN")
            and os.environ.get("DATAFORSEO_PASSWORD")
            and any(kw.get("search_volume") is not None for kw in keywords_data)
        )

    # Output report
    print_report(brand_summary, keywords_data, has_real_volumes)

    # Save CSV
    save_csv(keywords_data, args.output)
    print(f"\nSaved {len(keywords_data)} keywords → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
