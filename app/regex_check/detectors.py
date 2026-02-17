"""
detectors.py
Funkcje pomocnicze do analizy tekstu i wykrywania typów użytkowników.
"""
import re
from typing import Dict, List, Optional

from app.regex_check.patterns import (
    ADMIN_PATTERNS,
    ADMIN_USERNAME_PATTERN,
    BRAGGING_PATTERNS,
    BUILDER_PATTERN,
    HELPER_INDICATORS,
    HELPER_REPLY_RATIO,
    PROBLEM_INTENT_PATTERNS,
    PROBLEM_PATTERNS,
    PROBLEM_STATEMENT_PATTERN,
    QUESTION_INDICATORS,
    RECRUITER_PATTERNS,
    REJECT_KEYWORDS,
    REPLY_INDICATORS,
    SPAM_MESSAGE_THRESHOLD,
    SPAM_PATTERNS,
    TECH_SCORE_PATTERN,
    TECHNICAL_WHITELIST,
)


# ============ PODSTAWOWE TESTY TEKSTU ============


def has_technical_keywords(text: str) -> bool:
    return any(p.search(text) for p in TECHNICAL_WHITELIST)


def check_reject_keywords(text: str) -> Optional[str]:
    for p in REJECT_KEYWORDS:
        if p.search(text):
            return f"keyword: {p.pattern}"
    return None


def is_reply_pattern(text: str) -> bool:
    return any(p.search(text.strip()) for p in REPLY_INDICATORS)


def is_helper_pattern(text: str) -> bool:
    return any(p.search(text.strip()) for p in HELPER_INDICATORS)


def has_question_indicators(text: str) -> bool:
    return any(p.search(text) for p in QUESTION_INDICATORS)


def is_genuine_question(text: str) -> bool:
    if "?" in text:
        return True
    if has_technical_keywords(text) and any(
        w in text.lower() for w in ["help", "how", "anyone", "can someone"]
    ):
        return True
    return any(p.search(text) for p in PROBLEM_PATTERNS)


def has_problem_intent(text: str) -> bool:
    if not has_technical_keywords(text):
        return False
    return any(p.search(text) for p in PROBLEM_INTENT_PATTERNS)


def has_problem_statement(text: str) -> bool:
    return bool(PROBLEM_STATEMENT_PATTERN.search(text))


def is_too_short(text: str, min_words: int = 5) -> bool:
    return len(text.split()) < min_words


def is_obvious_spam(text: str) -> bool:
    return any(p.search(text) for p in SPAM_PATTERNS) or any(
        p.search(text) for p in BRAGGING_PATTERNS
    )


# ============ SCORING ============


def needs_help_score(msg: Dict, user_role: Optional[str] = None) -> float:
    """
    Skala 0.0–1.0.
    has_problem_intent i has_problem_statement nie sumują się –
    bierzemy max z obu (eliminuje double-counting).
    """
    score = 0.0
    text = msg.get("message", "")

    score += max(
        0.45 if has_problem_intent(text) else 0.0,
        0.35 if has_problem_statement(text) else 0.0,
    )
    if is_genuine_question(text):
        score += 0.2
    if has_technical_keywords(text):
        score += 0.15
    if TECH_SCORE_PATTERN.search(text):
        score += 0.10
    if len(text.split()) > 25:
        score += 0.10
    if BUILDER_PATTERN.search(text):
        score += 0.10
    if is_helper_pattern(text):
        score -= 0.25
    if is_reply_pattern(text):
        score -= 0.15
    if user_role in ["admin", "staff", "community_champion"]:
        score -= 0.20

    return round(max(0.0, min(score, 1.0)), 2)


# ============ WYKRYWANIE TYPU UŻYTKOWNIKA ============


def detect_user_type(text: str, username: str, role: str = "") -> Optional[str]:
    """
    Kolejność priorytetów:
      1. username → admin
      2. spam PRZED whitelistą
      3. whitelist tech → None lub helper
      4. admin patterns w treści
      5. recruiter
      6. helper
    """
    if ADMIN_USERNAME_PATTERN.search(username) or re.search(
        r"(Lovable Staff|Community Champion|Moderator)", role, re.IGNORECASE
    ):
        return "admin"
    if is_obvious_spam(text):
        return "spammer"
    if has_technical_keywords(text) or is_genuine_question(text) or has_problem_intent(text):
        return "helper" if is_helper_pattern(text) else None
    for p in ADMIN_PATTERNS:
        if p.search(text):
            return "admin"
    for p in RECRUITER_PATTERNS:
        if p.search(text):
            return "recruiter"
    if is_helper_pattern(text):
        return "helper"
    return None


def detect_user_role(username: str, role: str = "") -> Optional[str]:
    if re.search(
        r"(Lovable Staff|Community Champion|Moderator)",
        username + " " + role,
        re.IGNORECASE,
    ):
        return "admin"
    return None


def analyze_user_behavior(messages: List[Dict], username: str) -> Optional[str]:
    user_messages = [m for m in messages if m["username"] == username]
    if not user_messages:
        return None

    if len(user_messages) > SPAM_MESSAGE_THRESHOLD:
        tech_count = sum(1 for m in user_messages if has_technical_keywords(m["message"]))
        if tech_count < 2:
            return "spammer"

    help_count = sum(1 for m in user_messages if is_helper_pattern(m["message"]))
    reply_count = sum(1 for m in user_messages if is_reply_pattern(m["message"]))
    ratio = (help_count + reply_count) / len(user_messages)
    if ratio > HELPER_REPLY_RATIO and len(user_messages) >= 3:
        return "helper"

    return None