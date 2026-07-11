from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterable

from .catalog import CatalogItem, get_catalog, map_test_type
from .schemas import Recommendation


OFF_TOPIC_PATTERNS = [
    r"\blegal\b",
    r"\blawyer\b",
    r"\bgeneral hiring advice\b",
    r"\bhow do i hire\b",
    r"\binterview questions\b",
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\bresume\b",
    r"\bcv\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bignore previous instructions\b",
    r"\bprompt injection\b",
]

CLARIFY_PATTERNS = [
    r"\bi need an assessment\b",
    r"\bi need a test\b",
    r"\bneed an assessment\b",
    r"\bneed a solution\b",
    r"\bwhat should i use\b",
    r"\bhelp me pick\b",
]

AFFIRMATIVE_PATTERNS = [
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bperfect\b",
    r"\bthat works\b",
    r"\bsounds good\b",
    r"\bexactly\b",
    r"\bthat's what we need\b",
    r"\bgo ahead\b",
]

COMPARE_PATTERNS = [
    r"\bdifference between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bhow is .* different\b",
]

ROLE_WORDS = {
    "developer",
    "engineer",
    "manager",
    "analyst",
    "leader",
    "leadership",
    "executive",
    "director",
    "sales",
    "customer",
    "service",
    "contact",
    "cashier",
    "graduate",
    "technical",
    "support",
    "networking",
    "java",
    "python",
    "rust",
    "sql",
    "excel",
    "finance",
    "assessment",
}

REFINE_HINTS = {"also", "add", "instead", "actually", "include", "remove", "drop", "change", "update"}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

QUERY_BOOSTS = {
    "java": ["java"],
    "python": ["python"],
    "rust": ["rust"],
    "sql": ["sql"],
    "excel": ["excel"],
    "communication": ["communication", "interpersonal"],
    "communicate": ["communication", "interpersonal"],
    "stakeholder": ["communication", "interpersonal"],
    "stakeholders": ["communication", "interpersonal"],
    "leadership": ["leadership", "opq", "manager", "executive"],
    "personality": ["personality & behavior", "opq"],
    "behavior": ["personality & behavior", "opq"],
    "behaviour": ["personality & behavior", "opq"],
    "cognitive": ["ability & aptitude", "reasoning", "verify"],
    "ability": ["ability & aptitude", "reasoning", "verify"],
    "aptitude": ["ability & aptitude", "reasoning", "verify"],
    "reasoning": ["ability & aptitude", "reasoning", "verify"],
}

QUERY_BOOST_WEIGHTS = {
    "java": 0.85,
    "python": 0.75,
    "rust": 0.75,
    "sql": 0.7,
    "excel": 0.65,
    "communication": 0.45,
    "communicate": 0.45,
    "stakeholder": 0.45,
    "stakeholders": 0.45,
    "leadership": 0.35,
    "personality": 0.4,
    "behavior": 0.4,
    "behaviour": 0.4,
    "cognitive": 0.45,
    "ability": 0.45,
    "aptitude": 0.45,
    "reasoning": 0.45,
}


def _text(messages: Iterable[str]) -> str:
    return " \n".join(text.strip() for text in messages if text.strip())


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9+]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "you", "are", "with", "this", "that"}
    }


def _is_affirmative(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in AFFIRMATIVE_PATTERNS)


def _is_compare(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in COMPARE_PATTERNS)


def _is_off_topic(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in OFF_TOPIC_PATTERNS)


def _needs_clarification(text: str, history_text: str) -> bool:
    if any(token in text.lower() for token in REFINE_HINTS):
        return False
    if any(re.search(pattern, text, flags=re.I) for pattern in CLARIFY_PATTERNS):
        return True
    tokens = _meaningful_tokens(text)
    if len(tokens) <= 2:
        return True
    if not any(token in ROLE_WORDS for token in tokens):
        if len(tokens) <= 4:
            return True
    if re.search(r"\bleadership\b", text, flags=re.I) and not re.search(r"\b(selection|development|feedback|benchmark|hire|hiring)\b", history_text, flags=re.I):
        return True
    return False


def _compare_reply(item_a: CatalogItem, item_b: CatalogItem) -> str:
    parts = [
        f"{item_a.name} and {item_b.name} are different SHL products.",
    ]
    if item_a.test_type != item_b.test_type:
        parts.append(f"{item_a.name} is a {item_a.test_type} product, while {item_b.name} is a {item_b.test_type} product.")
    if item_a.duration and item_b.duration and item_a.duration != item_b.duration:
        parts.append(f"Duration: {item_a.name} is {item_a.duration}, while {item_b.name} is {item_b.duration}.")
    if item_a.keys and item_b.keys and item_a.keys != item_b.keys:
        parts.append(f"Coverage: {item_a.name} focuses on {', '.join(item_a.keys)}, while {item_b.name} focuses on {', '.join(item_b.keys)}.")
    if item_a.job_levels or item_b.job_levels:
        parts.append(
            f"Level fit: {item_a.name} supports {', '.join(item_a.job_levels) or 'no listed levels'}, while {item_b.name} supports {', '.join(item_b.job_levels) or 'no listed levels'}."
        )
    return " ".join(parts)


def _resolve_catalog_item(query: str) -> CatalogItem | None:
    return get_catalog().find_best(query)


def _build_messages_payload(user_messages: list[str], assistant_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise SHL assessment assistant. Only discuss SHL assessments from the supplied catalog context. "
                "Never mention products that are not in scope. Keep replies short and grounded."
            ),
        },
        {
            "role": "system",
            "content": assistant_context,
        },
    ]
    for text in user_messages:
        messages.append({"role": "user", "content": text})
    return json.dumps({"messages": messages})


def _generate_llm_reply(
    *,
    user_messages: list[str],
    reply_context: str,
    recommendations: list[Recommendation],
    fallback_reply: str,
) -> str:
    if not user_messages:
        return fallback_reply

    if not recommendations and _is_off_topic(user_messages[-1]):
        return fallback_reply

    catalog_context = "\n".join(
        [
            f"Latest user message: {user_messages[-1]}",
            f"Grounded assistant decision: {reply_context}",
            "If recommendations are present, mention them briefly without inventing anything.",
            "If this is a compare turn, explain only the differences already captured in the assistant decision.",
        ]
    )
    payload = _build_messages_payload(user_messages, catalog_context)

    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        body = json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": payload}],
                    }
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220},
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts).strip()
                if text:
                    return text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
            return fallback_reply

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        body = json.dumps(
            {
                "model": GROQ_MODEL,
                "messages": json.loads(payload)["messages"],
                "temperature": 0.2,
                "max_tokens": 220,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
                if text:
                    return text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
            return fallback_reply

    return fallback_reply


def _ranked_recommendations(query: str, limit: int = 5) -> list[Recommendation]:
    catalog = get_catalog()
    results = catalog.search(query, top_k=max(limit * 3, 15))
    if not results:
        return []

    query_tokens = _meaningful_tokens(query)
    query_text = query.lower()
    refined: list[tuple[CatalogItem, float]] = []
    for item, score in results:
        bonus = 0.0
        item_text = f"{item.name} {item.description} {' '.join(item.keys)} {' '.join(item.job_levels)}"
        item_tokens = _meaningful_tokens(item_text)
        overlap = len(query_tokens & item_tokens)
        bonus += min(overlap * 0.08, 0.25)
        for token, targets in QUERY_BOOSTS.items():
            if token in query_text and any(target in item_text.lower() for target in targets):
                bonus += QUERY_BOOST_WEIGHTS[token]
        if re.search(r"\b(personality|behavior|behaviour)\b", query, flags=re.I) and any(
            key.lower() == "personality & behavior" for key in item.keys
        ):
            bonus += 0.3
        if re.search(r"\b(cognitive|ability|aptitude|reasoning)\b", query, flags=re.I) and any(
            key.lower() in {"ability & aptitude", "biodata & situational judgment"} for key in item.keys
        ):
            bonus += 0.3
        if re.search(r"\b(coding|programming|developer|engineer|java|python|rust|sql|excel)\b", query, flags=re.I) and any(
            key.lower() == "knowledge & skills" for key in item.keys
        ):
            bonus += 0.18
        if re.search(r"\b(selection|hire|hiring|screen|shortlist)\b", query, flags=re.I) and item.job_levels:
            bonus += 0.05
        refined.append((item, score + bonus))

    refined.sort(key=lambda pair: pair[1], reverse=True)
    picked_items: list[CatalogItem] = []
    seen = set()
    for item, _score in refined:
        if item.link in seen:
            continue
        if item.is_packaged_solution:
            continue
        picked_items.append(item)
        seen.add(item.link)
        if len(picked_items) >= limit:
            break

    def has_type(test_type: str) -> bool:
        return any(map_test_type(item) == test_type for item in picked_items)

    def has_name_fragment(fragment: str) -> bool:
        return any(fragment in item.name.lower() for item in picked_items)

    query_lower = query.lower()
    if re.search(r"\b(personality|behavior|behaviour)\b", query_lower) and not has_type("P"):
        extra = (
            catalog.find_best("Occupational Personality Questionnaire OPQ32r")
            or catalog.find_best("OPQ32r")
            or catalog.find_best("OPQ")
        )
        if extra and map_test_type(extra) == "P":
            picked_items.insert(1 if picked_items else 0, extra)
    if re.search(r"\b(cognitive|ability|aptitude|reasoning)\b", query_lower) and not has_type("A"):
        extra = (
            catalog.find_best("SHL Verify Interactive G+")
            or catalog.find_best("Verify - G+")
            or catalog.find_best("Verify G+")
        )
        if extra and map_test_type(extra) == "A":
            picked_items.insert(1 if picked_items else 0, extra)
    if re.search(r"\b(stakeholder|stakeholders|communication|communicate)\b", query_lower) and not has_name_fragment("communication") and not has_name_fragment("interpersonal"):
        extra = (
            catalog.find_best("Business Communications")
            or catalog.find_best("Interpersonal Communications")
            or catalog.find_best("Business Communication (adaptive)")
        )
        if extra:
            picked_items.insert(2 if len(picked_items) >= 2 else len(picked_items), extra)

    deduped: list[CatalogItem] = []
    seen_links = set()
    for item in picked_items:
        if item.link in seen_links:
            continue
        seen_links.add(item.link)
        deduped.append(item)

    final_items = deduped[:limit]
    return [Recommendation(name=item.name, url=item.link, test_type=map_test_type(item)) for item in final_items]


def _build_search_query(messages: list[dict[str, str]]) -> str:
    user_texts = [message["content"] for message in messages if message.get("role") == "user"]
    if not user_texts:
        return ""
    latest = user_texts[-1]
    combined = _text(user_texts)
    if any(token in latest.lower() for token in REFINE_HINTS):
        return f"{latest} {latest} {combined}"
    return f"{latest} {combined}"


def handle_chat(messages: list[dict[str, str]]) -> tuple[str, list[Recommendation], bool]:
    if not messages:
        return (
            "Tell me the role, level, or skills you want to assess, and I’ll narrow it to SHL assessments.",
            [],
            False,
        )

    user_messages = [message["content"] for message in messages if message.get("role") == "user"]
    if not user_messages:
        return ("What role or assessment need should I map to the SHL catalog?", [], False)

    latest_user = user_messages[-1].strip()
    history_text = _text(user_messages[:-1])
    combined_text = _text(user_messages)

    if _is_off_topic(latest_user):
        return (
            "I can only help with SHL assessments from the SHL catalog. I can compare items, narrow a shortlist, or refine recommendations within that catalog.",
            [],
            False,
        )

    if _is_compare(latest_user):
        catalog = get_catalog()
        parts = re.split(r"\b(?:difference between|compare|vs\.?|versus)\b", latest_user, flags=re.I)
        compare_blob = parts[-1] if parts else latest_user
        candidates = [chunk.strip(" ?.,;:-") for chunk in re.split(r"\band\b|,|/", compare_blob) if chunk.strip(" ?.,;:-")]
        if len(candidates) < 2:
            recommendations = _ranked_recommendations(_build_search_query(messages), limit=2)
            if len(recommendations) >= 2:
                item_a = catalog.find_exact(recommendations[0].name)
                item_b = catalog.find_exact(recommendations[1].name)
                if item_a and item_b:
                    return (_compare_reply(item_a, item_b), [], False)
            return ("Which two SHL assessments should I compare?", [], False)

        item_a = _resolve_catalog_item(candidates[0])
        item_b = _resolve_catalog_item(candidates[1])
        if item_a and item_b:
            fallback_reply = _compare_reply(item_a, item_b)
            llm_reply = _generate_llm_reply(
                user_messages=user_messages,
                reply_context=fallback_reply,
                recommendations=[],
                fallback_reply=fallback_reply,
            )
            return (llm_reply, [], False)
        return ("I could not confidently identify both assessments. Please give the exact SHL product names.", [], False)

    if _needs_clarification(latest_user, history_text):
        if re.search(r"\bleadership\b", combined_text, flags=re.I):
            return ("Who is this for, and is it for selection or development?", [], False)
        return ("What role, level, or skill area should I use to narrow the SHL catalog?", [], False)

    query = _build_search_query(messages)
    recommendations = _ranked_recommendations(query, limit=5)
    if not recommendations:
        return ("I could not find a grounded SHL shortlist yet. Give me the role, level, or key skills, and I’ll narrow it within the catalog.", [], False)

    fallback_reply = f"Got it. Here are {len(recommendations)} SHL assessments from the catalog that fit this need."
    llm_reply = _generate_llm_reply(
        user_messages=user_messages,
        reply_context=fallback_reply,
        recommendations=recommendations,
        fallback_reply=fallback_reply,
    )

    if _is_affirmative(latest_user):
        reply = llm_reply if llm_reply else f"That matches. Here are {len(recommendations)} SHL assessments from the catalog that fit this need."
        return (reply, recommendations, True)

    reply = llm_reply
    return (reply, recommendations, False)
