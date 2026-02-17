"""
blacklist_utils.py
Narzędzia CLI do zarządzania blacklistą: podgląd, eksport, ręczna edycja.
"""
from typing import Dict, List

from app.regex_check.blacklist import BLACKLIST


def manually_add_to_blacklist(username: str, category: str, reason: str = "manually added"):
    BLACKLIST.add_user(username, category, reason)


def manually_remove_from_blacklist(username: str):
    BLACKLIST.remove_user(username)


def show_blacklist():
    blacklist = BLACKLIST.export_list()
    if not blacklist:
        print("\n📝 Blacklista jest pusta")
        return

    print("\n" + "=" * 60)
    print("BLACKLISTA UŻYTKOWNIKÓW")
    print("=" * 60)
    stats = BLACKLIST.get_stats()
    print(f"\n📊 Statystyki: {len(blacklist)} użytkowników")
    for category, count in stats.items():
        print(f"   - {category}: {count}")
    print("\n" + "=" * 60)

    categories: Dict[str, List] = {}
    for item in blacklist:
        categories.setdefault(item["category"], []).append(item)

    for category, users in categories.items():
        print(f"\n🚫 {category.upper()}:")
        for user in users:
            print(f"   - {user['username']} (dodano: {user['added_date'][:10]})")
            if user.get("reason"):
                print(f"     └─ {user['reason'][:60]}...")


def export_blacklist_txt(filename: str = "blacklist_export.txt"):
    blacklist = BLACKLIST.export_list()
    with open(filename, "w", encoding="utf-8") as f:
        for item in blacklist:
            f.write(f"{item['username']}\t{item['category']}\t{item['added_date']}\n")
    print(f"✅ Blacklista wyeksportowana do {filename}")


def show_candidates(candidates: list):
    print("LISTA KANDYDATÓW:")
    print("-" * 80)
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c['username']}")
        print(f"   [{c['timestamp']}] {c['message'][:70]}...")
        print()