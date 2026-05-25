import json
import re
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ==== CONFIG ====
LLM_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3-4b-instruct"  # change if needed

ALLOWED_TAGS = [
    "historic", "cultural", "religious-heritage", "beach", "coastal", "island", "urban",
    "nightlife", "food", "shopping", "luxury", "affordable", "budget", "family-friendly",
    "romantic", "nature", "adventure", "relaxation", "touristic", "crowded", "quiet",
    "capital", "modern", "traditional", "business", "party", "backpacking", "desert",
    "mountain", "forest", "lake", "river", "winter-sports", "ski", "diving", "water-sports",
    "wellness", "spa", "shopping-luxury", "shopping-budget", "street-food", "fine-dining",
    "art", "architecture", "museums", "festivals", "events", "pilgrimage", "spiritual",
    "photography", "scenic", "remote", "wildlife", "eco-tourism",
]

ALLOWED_TAGS_TEXT = ", ".join(ALLOWED_TAGS)

# ==== SYSTEM PROMPT ====
SYSTEM_PROMPT = """
You are a strict preference extraction agent for a travel planner.

Your job:
Extract structured user preferences from conversation and map them into valid travel tags.

ALLOWED TAGS (use ONLY these, no new tags):
__ALLOWED_TAGS__

You MUST extract:
- budget_total (numeric if given)
- destination (city/country if mentioned)
- travel_style (array of styles)
- duration_days (integer days if mentioned)
- constraints (array of strings)
- tags (array of 3 to 8 tags from ALLOWED TAGS)
- expense_tag based on budget_total and duration_days

Rules:
- Ask follow-up questions if missing important info
- Be concise
- NEVER explain yourself
- If budget_total and duration_days are both known, compute:
    - daily_budget = budget_total / duration_days
    - expense_tag:
        - daily_budget < 80 => "budget"
        - 80 <= daily_budget < 200 => "affordable"
        - daily_budget >= 200 => "luxury"
- If numeric data is missing, infer expense_tag from language:
    - low/cheap => "budget"
    - medium/moderate => "affordable"
    - high/luxury/expensive => "luxury"

When you have enough information, output EXACTLY:

STOP
{
    "budget": {
        "raw": "...",
        "total": 0,
        "currency": "..."
    },
  "destination": "...",
    "travel_style": ["..."],
    "duration_days": 0,
    "constraints": ["..."],
    "tags": ["..."],
    "expense_tag": "budget|affordable|luxury",
    "daily_budget": 0
}

If not enough info:
- Ask the next best question
- DO NOT output STOP
"""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("__ALLOWED_TAGS__", ALLOWED_TAGS_TEXT)

# ==== MEMORY ====
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def call_llm(messages):
    response = requests.post(
        LLM_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3
        }
    )
    return response.json()["choices"][0]["message"]["content"]


def _extract_first_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"-?\d+(?:[\.,]\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def _normalize_duration_days(value):
    number = _extract_first_number(value)
    if number is None or number <= 0:
        return None
    return int(round(number))


def _normalize_budget(budget_obj):
    raw = ""
    currency = ""
    total = None

    if isinstance(budget_obj, dict):
        raw = str(budget_obj.get("raw", ""))
        currency = str(budget_obj.get("currency", ""))
        total = _extract_first_number(budget_obj.get("total"))
        if total is None:
            total = _extract_first_number(raw)
    else:
        raw = str(budget_obj or "")
        total = _extract_first_number(raw)

    return {
        "raw": raw,
        "total": total,
        "currency": currency,
    }


def _infer_expense_tag_from_text(text):
    t = (text or "").lower()
    if any(k in t for k in ["luxury", "expensive", "high"]):
        return "luxury"
    if any(k in t for k in ["medium", "moderate", "mid"]):
        return "affordable"
    if any(k in t for k in ["budget", "cheap", "low"]):
        return "budget"
    return None


def _compute_expense_tag(budget, duration_days):
    total = budget.get("total")
    if total is not None and duration_days:
        daily_budget = total / duration_days
        if daily_budget < 80:
            return "budget", daily_budget
        if daily_budget < 200:
            return "affordable", daily_budget
        return "luxury", daily_budget

    inferred = _infer_expense_tag_from_text(budget.get("raw", ""))
    return inferred, None


def normalize_preferences(data):
    budget = _normalize_budget(data.get("budget"))
    duration_days = _normalize_duration_days(data.get("duration_days"))

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip() in ALLOWED_TAGS]

    travel_style = data.get("travel_style", [])
    if not isinstance(travel_style, list):
        travel_style = [str(travel_style)] if travel_style else []
    travel_style = [str(s).strip() for s in travel_style if str(s).strip()]

    constraints = data.get("constraints", [])
    if not isinstance(constraints, list):
        constraints = [str(constraints)] if constraints else []
    constraints = [str(c).strip() for c in constraints if str(c).strip()]

    expense_tag, daily_budget = _compute_expense_tag(budget, duration_days)
    if expense_tag and expense_tag not in tags:
        tags.append(expense_tag)

    return {
        "destination": str(data.get("destination", "")).strip(),
        "budget": budget,
        "duration_days": duration_days,
        "daily_budget": daily_budget,
        "expense_tag": expense_tag,
        "travel_style": travel_style,
        "constraints": constraints,
        "tags": tags,
    }


# ==== ROUTE ====
@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    user_input = request.json.get("message")

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    reply = call_llm(conversation_history)

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    # ==== STOP DETECTION ====
    if reply.strip().startswith("STOP"):
        try:
            json_part = reply.split("STOP", 1)[1].strip()
            data = normalize_preferences(json.loads(json_part))

            # Reset memory for next session
            conversation_history = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            return jsonify({
                "status": "complete",
                "data": data
            })

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Failed to parse STOP output",
                "raw": reply
            })

    return jsonify({
        "status": "ongoing",
        "reply": reply
    })


# ==== RUN ====
if __name__ == "__main__":
    app.run(port=5001, debug=True)