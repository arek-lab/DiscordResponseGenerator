"""
blacklist.py
Zarządzanie blacklistą użytkowników (zapis/odczyt JSON).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.regex_check.patterns import BLACKLIST_FILE

logger = logging.getLogger(__name__)


class UserBlacklist:
    def __init__(self, filepath: Path = BLACKLIST_FILE):
        self.filepath = filepath
        self.blacklisted_users: Dict[str, Dict] = {}
        self._load_from_file()

    def _load_from_file(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.blacklisted_users = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Blacklist file corrupted, starting fresh.")
                self.blacklisted_users = {}
        else:
            self.blacklisted_users = {}

    def _save_to_file(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.blacklisted_users, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Błąd zapisu blacklisty: {e}")

    def add_user(self, username: str, category: str, reason: str = "auto-detected"):
        if username not in self.blacklisted_users:
            self.blacklisted_users[username] = {
                "category": category,
                "added_date": datetime.now().isoformat(),
                "reason": reason,
            }
            self._save_to_file()

    def is_blacklisted(self, username: str) -> bool:
        return username in self.blacklisted_users

    def get_category(self, username: str) -> Optional[str]:
        return self.blacklisted_users.get(username, {}).get("category")

    def get_info(self, username: str) -> Optional[Dict]:
        return self.blacklisted_users.get(username)

    def remove_user(self, username: str):
        if username in self.blacklisted_users:
            del self.blacklisted_users[username]
            self._save_to_file()

    def export_list(self) -> List[Dict]:
        return [{"username": u, **info} for u, info in self.blacklisted_users.items()]

    def get_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for info in self.blacklisted_users.values():
            cat = info["category"]
            stats[cat] = stats.get(cat, 0) + 1
        return stats


# Singleton używany w całej aplikacji
BLACKLIST = UserBlacklist()