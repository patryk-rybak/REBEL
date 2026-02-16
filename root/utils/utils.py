"""
Text normalization and matching helpers. Used for key phrase extraction and similarity checks.
"""

import re
import unicodedata
from typing import List


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z][a-z\-']{1,}", normalize_text(s))


def extract_key_phrase_from_answer(answer: str) -> str:
    m = re.search(r"[\"“”'`]+(.+?)[\"“”'`]+", answer)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"\b(is|are|was|were)\s+(.+?)([.?!]|$)", answer, flags=re.I)
    if m2:
        candidate = m2.group(2).strip()
        candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.I).strip()
        return candidate

    caps = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b", answer)
    if caps:
        return caps[-1]

    return answer.strip()


def levenshtein_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            temp = dp[j]
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[lb]


def similar_enough(a: str, b: str, threshold: float) -> bool:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    dist = levenshtein_distance(a_norm, b_norm)
    max_len = max(len(a_norm), len(b_norm))
    if max_len == 0:
        return True
    similarity = 1.0 - dist / max_len
    return similarity >= threshold


def words_lower(s: str) -> List[str]:
    return re.findall(r"[a-z]+", normalize_text(s))


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z][a-z\-']{1,}", normalize_text(s))


def is_name_like(phrase: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", phrase)
    if len(tokens) < 2:
        return False
    caps = sum(1 for t in tokens if re.match(r"^[A-Z][a-z]+$", t))
    return caps >= 2


def has_contiguous_tokens(reply: str, tokens: List[str]) -> bool:
    rtoks = words_lower(reply)
    n = len(tokens)
    if n == 0:
        return False
    for i in range(0, max(0, len(rtoks) - n + 1)):
        if rtoks[i : i + n] == tokens:
            return True
    return False
