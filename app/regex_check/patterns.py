"""
patterns.py
Wszystkie wyrażenia regularne i stałe konfiguracyjne.
"""
import re
from pathlib import Path

# ============ TECHNICAL WHITELIST ============

TECHNICAL_WHITELIST = [
    re.compile(r"\bsupabase\b", re.IGNORECASE),
    re.compile(r"\bdatabase\b", re.IGNORECASE),
    re.compile(r"\bpostgres\b", re.IGNORECASE),
    re.compile(r"\bapi\b", re.IGNORECASE),
    re.compile(r"\berror\b.*\blog\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+get\b", re.IGNORECASE),
    re.compile(r"\bcache\s+(loop|error|issue)\b", re.IGNORECASE),
    re.compile(r"\bbiometric\b.*\blogin\b", re.IGNORECASE),
    re.compile(r"\brpc\s+function", re.IGNORECASE),
    re.compile(r"\bdataflow\b", re.IGNORECASE),
    re.compile(r"\bbackup\b.*\bdatabase\b", re.IGNORECASE),
    re.compile(r"\bauth\b.*\b(error|issue|problem)\b", re.IGNORECASE),
    re.compile(r"\breact\b.*\b(native|capacitor)\b", re.IGNORECASE),
    re.compile(r"\bshopify\b", re.IGNORECASE),
    re.compile(r"\bstripe\b", re.IGNORECASE),
    re.compile(r"\brealtime\b", re.IGNORECASE),
    re.compile(r"\breplication\s+slot\b", re.IGNORECASE),
]

# ============ REJECT / SPAM / BRAGGING / RECRUITER ============

REJECT_KEYWORDS = [
    re.compile(r"\blovable\b.*(scam|garbage|trash|bullshit)", re.IGNORECASE),
    re.compile(r"\bcredits?\b.*(stole|stealing|robbed)", re.IGNORECASE),
    re.compile(r"\bmake\s+money\s+(fast|easy|now)\b", re.IGNORECASE),
    re.compile(r"\bget\s+rich\s+quick\b", re.IGNORECASE),
]

ADMIN_PATTERNS = [
    re.compile(r"Ikona roli,?\s+(Lovable Staff|Community Champion|Moderator)", re.IGNORECASE),
    re.compile(r"^(AdminBot|Lovable\s+Mod|Dyno)\b", re.IGNORECASE),
    re.compile(r"@(everyone|here)", re.IGNORECASE),
]

SPAM_PATTERNS = [
    re.compile(r"\bcheck\s+out\s+my\b.*\b(website|tool|app)\b.*\b(now|today|link)\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\s+(to\s+)?(my|our)\b.*\b(channel|newsletter)\b", re.IGNORECASE),
    re.compile(r"\bfollow\s+me\s+(on|at)\b.*\b(instagram|twitter|youtube)\b", re.IGNORECASE),
    re.compile(r"\blink\s+in\s+bio\b", re.IGNORECASE),
    re.compile(r"(discord\.gg|bit\.ly|t\.me)/\w+.*\bjoin\b", re.IGNORECASE),
    re.compile(r"\b(buy|purchase|order)\s+now\b.*\blimited\b", re.IGNORECASE),
    re.compile(r"\bfree\s+(trial|download)\b.*\b(today only|limited time)\b", re.IGNORECASE),
    re.compile(r"\bjoin\s+my\s+launch\b.*\b(live|going live|hours?)\b", re.IGNORECASE),
    re.compile(r"\bproduct\s+hunt\b.*\b(live|launch|vote)\b", re.IGNORECASE),
]

BRAGGING_PATTERNS = [
    re.compile(r"\bmade\s+\$?\d+k\b.*\bthanks\s+to\s+(lovable|the\s+team)\b", re.IGNORECASE),
    re.compile(r"\$\d+M\s+a\s+month\b.*\bdownloads\b", re.IGNORECASE),
]

RECRUITER_PATTERNS = [
    re.compile(r"\b(hiring|recruiting)\b.*\b(developer|engineer|designer)\b", re.IGNORECASE),
    re.compile(r"\bwe'?re\s+looking\s+for\b.*\b(developer|engineer|team member)\b", re.IGNORECASE),
    re.compile(r"\bjoin\s+(our|my)\s+team\b", re.IGNORECASE),
    re.compile(r"\bapply\s+now\b.*\b(job|position|role)\b", re.IGNORECASE),
]

# ============ HELPER / REPLY / QUESTION ============

HELPER_INDICATORS = [
    re.compile(r"^(sure|okay|yes),?\s+(dm|i can help)", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+(help|check|look)\b", re.IGNORECASE),
    re.compile(r"\bhave\s+a\s+look\b", re.IGNORECASE),
    re.compile(r"\bsend\s+me\s+a\s+dm\b", re.IGNORECASE),
    re.compile(r"\bdm\s*!\s*$", re.IGNORECASE),
    re.compile(r"\bi\s+have\s+had\s+that\s+issue\b", re.IGNORECASE),
    re.compile(r"\byou\s+need\s+to\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+troubles?\s+(are\s+you|do\s+you)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(issue|problem)\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bi\s+(can\s+help|helped|have\s+done)\s+(with\s+)?(that|this|it)\b", re.IGNORECASE),
    re.compile(r"\bi\s+implemented\s+(a\s+part\s+of\s+)?functionality\b", re.IGNORECASE),
]

REPLY_INDICATORS = [
    re.compile(r"^sure,?\s", re.IGNORECASE),
    re.compile(r"^okay,?\s", re.IGNORECASE),
    re.compile(r"^yes,?\s", re.IGNORECASE),
    re.compile(r"^no,?\s", re.IGNORECASE),
    re.compile(r"^that'?s\b", re.IGNORECASE),
    re.compile(r"\bdm!?\s*$", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+know\b", re.IGNORECASE),
    re.compile(r"^cool\b", re.IGNORECASE),
    re.compile(r"^nice\b", re.IGNORECASE),
]

QUESTION_INDICATORS = [
    re.compile(r"\?"),
    re.compile(r"\bhelp\b", re.IGNORECASE),
    re.compile(r"\bneed\b", re.IGNORECASE),
    re.compile(r"\banyone\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\b", re.IGNORECASE),
    re.compile(r"\bwhere\b", re.IGNORECASE),
    re.compile(r"\bwhat\b", re.IGNORECASE),
    re.compile(r"\bwhy\b", re.IGNORECASE),
    re.compile(r"\bcan\s+someone\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+anyone\b", re.IGNORECASE),
]

# ============ PROBLEM PATTERNS (wewnętrzne) ============

PROBLEM_PATTERNS = [
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bissue\b", re.IGNORECASE),
    re.compile(r"\bproblem\b", re.IGNORECASE),
    re.compile(r"\bnot\s+working\b", re.IGNORECASE),
    re.compile(r"\bcan'?t\b", re.IGNORECASE),
    re.compile(r"\bdoes'?n'?t\s+work\b", re.IGNORECASE),
]

PROBLEM_INTENT_PATTERNS = [
    re.compile(r"\bi('?m| am)\s+(stuck|blocked|confused|lost)\b", re.IGNORECASE),
    re.compile(r"\b(can'?t|cannot|couldn'?t)\b", re.IGNORECASE),
    re.compile(r"\bdoesn'?t\s+work\b", re.IGNORECASE),
    re.compile(r"\bnot\s+working\b", re.IGNORECASE),
    re.compile(r"\bissue\b", re.IGNORECASE),
    re.compile(r"\bproblem\b", re.IGNORECASE),
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bkeeps\s+(failing|loading|breaking)\b", re.IGNORECASE),
    re.compile(r"\bsession\s+expired\b", re.IGNORECASE),
    re.compile(r"\bno\s+solution\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+anyone\s+else\b", re.IGNORECASE),
    re.compile(r"\bstuck\b", re.IGNORECASE),
    re.compile(r"\bdown\b.*\b(due to|because of)\b", re.IGNORECASE),
]

PROBLEM_STATEMENT_PATTERN = re.compile(
    r"\b(broken|messed up|not working|issue|problem|stuck|failed|error|timeout|down)\b",
    re.IGNORECASE,
)

TECH_SCORE_PATTERN = re.compile(
    r"\b(api|database|db|auth|supabase|rpc|cache|session|logs|domain|backup)\b",
    re.IGNORECASE,
)

BUILDER_PATTERN = re.compile(
    r"\b(my|our)\s+(project|app|application|website)\b",
    re.IGNORECASE,
)

ADMIN_USERNAME_PATTERN = re.compile(
    r"(Lovable Staff|Community Champion|Moderator|AdminBot|Dyno)",
    re.IGNORECASE,
)

# ============ PROGI / STAŁE ============

SPAM_MESSAGE_THRESHOLD = 8
HELPER_REPLY_RATIO = 0.7
BLACKLIST_FILE = Path(__file__).parent / "blacklist.json"