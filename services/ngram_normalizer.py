# services/ngram_normalizer.py
from pathlib import Path
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class NGramLanguageModel:
    def __init__(self, bi_path: str, tri_path: str, four_path: str):
        self.bigrams = set()
        self.trigrams = set()
        self.fourgrams = set()
        self.vocab = set()
        self._load_ngrams(bi_path, tri_path, four_path)

    def _load_file(self, path: str) -> list:
        p = Path(path)
        if not p.exists():
            logger.warning(f"N-gram file not found: {path}")
            return []
        with open(p, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    def _load_ngrams(self, bi_path: str, tri_path: str, four_path: str):
        for line in self._load_file(bi_path):
            toks = line.split()
            if len(toks) == 2:
                self.bigrams.add(tuple(toks))
                self.vocab.update(toks)

        for line in self._load_file(tri_path):
            toks = line.split()
            if len(toks) == 3:
                self.trigrams.add(tuple(toks))
                self.vocab.update(toks)

        for line in self._load_file(four_path):
            toks = line.split()
            if len(toks) == 4:
                self.fourgrams.add(tuple(toks))
                self.vocab.update(toks)

        logger.info(
            f"Loaded n-grams: {len(self.bigrams)} bi, "
            f"{len(self.trigrams)} tri, {len(self.fourgrams)} four; "
            f"vocab={len(self.vocab)}"
        )

    def score(self, tokens: List[str]) -> int:
        score = 0
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i+1]) in self.bigrams:
                score += 1
        for i in range(len(tokens) - 2):
            if (tokens[i], tokens[i+1], tokens[i+2]) in self.trigrams:
                score += 2
        for i in range(len(tokens) - 3):
            if (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]) in self.fourgrams:
                score += 3
        return score


class VNSpellCorrector:
    def __init__(self, lm: NGramLanguageModel, max_edit_distance: int = 1):
        self.lm = lm
        self.max_edit = max_edit_distance

    def _edit_distance(self, s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if abs(len(s1) - len(s2)) > self.max_edit:
            return self.max_edit + 1
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[m][n]

    def _candidates(self, word: str) -> list:
        if word in self.lm.vocab:
            return [word]
        cands = []
        lw = len(word)
        for v in self.lm.vocab:
            if abs(len(v) - lw) <= self.max_edit:
                if self._edit_distance(word, v) <= self.max_edit:
                    cands.append(v)
        return cands[:20]

    def correct_sentence(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        tokens = text.split()
        if not tokens:
            return text

        for i, w in enumerate(tokens):
            if w in self.lm.vocab:
                continue
            cands = self._candidates(w)
            if not cands:
                continue

            best = w
            best_score = -1
            for cand in cands:
                new_tokens = tokens.copy()
                new_tokens[i] = cand
                s = self.lm.score(new_tokens)
                if s > best_score:
                    best_score = s
                    best = cand
            tokens[i] = best

        return " ".join(tokens)
