"""
filters.py
Główna logika filtrowania wiadomości oraz pipeline przetwarzania.
"""
import copy
from typing import Dict, List, Tuple

from app.regex_check.blacklist import BLACKLIST
from app.regex_check.detectors import (
    analyze_user_behavior,
    check_reject_keywords,
    detect_user_role,
    detect_user_type,
    has_problem_intent,
    has_question_indicators,
    has_technical_keywords,
    needs_help_score,
    is_too_short,
)
from app.regex_check.parser import parse_discord_messages


# ============ BLACKLIST UPDATE ============


def detect_and_update_blacklist(messages: List[Dict]) -> Dict[str, str]:
    detected: Dict[str, str] = {}
    for msg in messages:
        username = msg["username"]
        role = msg.get("role", "")
        if BLACKLIST.is_blacklisted(username):
            continue
        user_type = detect_user_type(msg["message"], username, role)
        if user_type in ["admin", "spammer", "recruiter"]:
            detected[username] = user_type
            BLACKLIST.add_user(username, user_type, f"Pattern detected: {msg['message'][:50]}...")

    for username in {m["username"] for m in messages}:
        if BLACKLIST.is_blacklisted(username):
            continue
        behavior = analyze_user_behavior(messages, username)
        if behavior == "spammer":
            count = sum(1 for m in messages if m["username"] == username)
            detected[username] = behavior
            BLACKLIST.add_user(username, behavior, f"Behavior: {count} messages")

    return detected


# ============ FILTROWANIE ============


def filter_messages(messages: List[Dict]) -> List[Dict]:
    """
    Zwraca NOWĄ listę (nie mutuje wejścia).
    """
    messages = copy.deepcopy(messages)

    detect_and_update_blacklist(messages)

    print(f"DEBUG filter_messages: przetwarzam {len(messages)} wiadomości")

    for msg in messages:
        username = msg["username"]
        text = msg["message"]
        msg["skip"] = False
        msg["auto_reject_reason"] = None

        # CHECK 0: Blacklisted (nie blokuj helperów)
        if BLACKLIST.is_blacklisted(username):
            if BLACKLIST.get_category(username) != "helper":
                msg["skip"] = True
                msg["auto_reject_reason"] = f"blacklisted_user:{BLACKLIST.get_category(username)}"
                continue

        # CHECK 1: Reject keywords
        reject_reason = check_reject_keywords(text)
        if reject_reason:
            msg["skip"] = True
            msg["auto_reject_reason"] = reject_reason
            continue

        # CHECK 2: Przekazane wiadomości
        if msg["is_forwarded"]:
            msg["skip"] = True
            msg["auto_reject_reason"] = "forwarded_message"
            continue

        # CHECK 3: Za krótka bez pytania/tech
        if is_too_short(text):
            if (
                not has_question_indicators(text)
                and not has_technical_keywords(text)
                and not has_problem_intent(text)
            ):
                msg["skip"] = True
                msg["auto_reject_reason"] = "too_short_no_question"
                continue

        # CHECK 4: Ogólny komentarz
        if not has_question_indicators(text) and not is_reply_pattern(text):
            if has_problem_intent(text):
                continue
            if not has_technical_keywords(text):
                msg["skip"] = True
                msg["auto_reject_reason"] = "general_comment"
                continue

    return messages


def get_candidates(messages: List[Dict]) -> List[Dict]:
    return [msg for msg in messages if not msg["skip"]]


# ============ PIPELINE ============


def process_filters(text: str) -> Tuple[List[Dict], List[Dict]]:
    messages = parse_discord_messages(text)

    print(f"\n{'=' * 80}")
    print(f"📊 STATYSTYKI FILTROWANIA")
    print(f"{'=' * 80}")
    print(f"Liczba wszystkich wiadomości: {len(messages)}")

    filtered = filter_messages(messages)

    rejection_reasons: Dict[str, int] = {}
    for m in filtered:
        if m["skip"]:
            reason = m["auto_reject_reason"]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    print(f"\n📉 Odrzucone: {sum(rejection_reasons.values())}")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {reason}: {count}")

    candidates = get_candidates(filtered)
    print(f"\n✅ Zaakceptowane: {len(candidates)}")
    print(f"{'=' * 80}\n")

    return candidates, filtered


def process_messages(text: str) -> Tuple[List[Dict], List[Dict]]:
    candidates, all_messages = process_filters(text)
    for msg in candidates:
        role = detect_user_role(msg["username"], msg.get("role", ""))
        msg["needs_help_score"] = needs_help_score(msg, role)
    return candidates, all_messages


# ============ IMPORT BRAKUJĄCEJ FUNKCJI ============
# is_reply_pattern jest potrzebna w filter_messages
from app.regex_check.detectors import is_reply_pattern  # noqa: E402 (celowy import na dole)