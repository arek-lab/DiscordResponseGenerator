"""
parser.py
Parsowanie surowego tekstu Discorda do listy słowników.
"""
import re
from typing import Dict, List


def parse_discord_messages(raw_text: str) -> List[Dict]:
    raw_text = raw_text.replace("\u2060", "")
    lines = [l.strip() for l in raw_text.split("\n")]
    messages = []

    TIMESTAMP_PATTERN = re.compile(
        r"(?:.*?—\s+)?(?:(?:Wczoraj|Dzisiaj|Dziś|Yesterday|Today)\s+(?:o\s+)?)?(\d{1,2}:\d{2}(?:\s*[AP]M)?)$",
        re.IGNORECASE,
    )

    def is_meta_line(line: str) -> bool:
        low = line.lower()
        # Linia z "ikona roli" + timestamp = nagłówek admina, NIE śmieć
        if "ikona roli" in low and re.search(r"\d{1,2}:\d{2}", line):
            return False
        return any(x in low for x in ["ikona roli", "shared with me", "edycja", "odpowiedz"])

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        ts_match = TIMESTAMP_PATTERN.search(line)

        if ts_match:
            timestamp = ts_match.group(1)
            username = "Nieznany"
            found_role = ""

            # Wariant A: "username — HH:MM" (jednolinowy, nie-admin)
            if " — " in line and not line.lower().startswith("ikona roli"):
                potential_user = line.split(" — ")[0].strip()
                if potential_user and not is_meta_line(potential_user):
                    username = potential_user

            # Wariant B: "Ikona roli, Rola — HH:MM" (admin dwuliniowy)
            elif line.lower().startswith("ikona roli"):
                found_role = line
                for j in range(i - 1, -1, -1):
                    prev = lines[j]
                    if not prev:
                        continue
                    if not is_meta_line(prev):
                        username = prev
                        if messages and messages[-1]["message"].endswith(username):
                            messages[-1]["message"] = messages[-1]["message"][: -len(username)].strip()
                        break

            # Wariant C: "— HH:MM" (username w poprzedniej linii, bez roli)
            elif line.startswith("—"):
                for j in range(i - 1, -1, -1):
                    prev = lines[j]
                    if not prev:
                        continue
                    if not is_meta_line(prev):
                        username = prev
                        if messages and messages[-1]["message"].endswith(username):
                            messages[-1]["message"] = messages[-1]["message"][: -len(username)].strip()
                        break

            messages.append(
                {
                    "username": username,
                    "timestamp": timestamp,
                    "message": "",
                    "has_images": False,
                    "is_forwarded": False,
                    "role": found_role,
                }
            )
            i += 1
            continue

        if messages:
            curr = messages[-1]
            l_low = line.lower()
            if l_low == "obraz":
                curr["has_images"] = True
            elif "przekazano dalej" in l_low:
                curr["is_forwarded"] = True
            elif not is_meta_line(line) and not re.match(r"^[_\-]{3,}$", line):
                curr["message"] = curr["message"] + "\n" + line if curr["message"] else line
        i += 1

    final_data = []
    for m in messages:
        m["message"] = re.sub(r" +", " ", m["message"].strip())
        if m["message"] or m["has_images"]:
            final_data.append(m)

    return final_data