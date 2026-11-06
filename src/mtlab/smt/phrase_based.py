from __future__ import annotations

import math
import pickle
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from itertools import zip_longest


_A3_HEADER = re.compile(r"^# Sentence pair\s*\((\d+)\)")
_A3_GROUP = re.compile(r"(\S+)\s+\(\{\s*([^{}]*?)\s*\}\)")


def parse_a3_record(header: str, source_line: str, alignment_line: str):
    match = _A3_HEADER.search(header)
    if not match:
        raise ValueError(f"Invalid A3 record header: {header.strip()}")
    sentence_no = int(match.group(1))
    source_tokens = source_line.strip().split()
    groups = _A3_GROUP.findall(alignment_line)
    if not groups:
        raise ValueError(f"No alignment groups in A3 sentence {sentence_no}")

    target_tokens: list[str] = []
    links: set[tuple[int, int]] = set()
    target_pos = 0
    for word, positions in groups:
        if word == "NULL":
            continue
        target_tokens.append(word)
        for value in positions.split():
            source_pos = int(value)
            if source_pos == 0:
                continue
            if not 1 <= source_pos <= len(source_tokens):
                raise ValueError(f"A3 source index out of range in sentence {sentence_no}")
            links.add((source_pos - 1, target_pos))
        target_pos += 1
    return sentence_no, source_tokens, target_tokens, links


def iter_a3(path: str | Path) -> Iterator[tuple[int, list[str], list[str], set[tuple[int, int]]]]:
    with Path(path).open(encoding="utf-8") as file:
        while True:
            header = file.readline()
            if not header:
                return
            if not header.strip():
                continue
            source = file.readline()
            alignment = file.readline()
            if not source or not alignment:
                raise ValueError(f"Incomplete A3 record in {path}")
            yield parse_a3_record(header, source, alignment)


def grow_diag_final_and(forward: set[tuple[int, int]], reverse: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Moses-style grow-diag-final-and symmetrization in (src,tgt) coordinates."""
    intersection = forward & reverse
    union = forward | reverse
    result = set(intersection)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),    (1, 1),
    ]

    changed = True
    while changed:
        changed = False
        for i, j in sorted(union - result):
            if not any((i + di, j + dj) in result for di, dj in neighbors):
                continue
            source_aligned = any(si == i for si, _ in result)
            target_aligned = any(tj == j for _, tj in result)
            if not source_aligned or not target_aligned:
                result.add((i, j))
                changed = True

    # Final-and: add remaining links only when both endpoints are unaligned.
    for i, j in sorted(union - result):
        source_aligned = any(si == i for si, _ in result)
        target_aligned = any(tj == j for _, tj in result)
        if not source_aligned and not target_aligned:
            result.add((i, j))
    return result


def phrase_pairs_from_alignment(
    source: list[str],
    target: list[str],
    links: set[tuple[int, int]],
    max_phrase_len: int = 7,
) -> Iterator[tuple[tuple[str, ...], tuple[str, ...], tuple[int, int, int, int]]]:
    """Extract all phrase pairs consistent with a word alignment."""
    src_len, tgt_len = len(source), len(target)
    src_aligned = {i for i, _ in links}
    tgt_aligned = {j for _, j in links}

    for i1 in range(src_len):
        for i2 in range(i1, min(src_len, i1 + max_phrase_len)):
            target_positions = [j for i, j in links if i1 <= i <= i2]
            if not target_positions:
                continue
            j1, j2 = min(target_positions), max(target_positions)

            # A source word outside [i1,i2] must not align inside [j1,j2].
            if any((i < i1 or i > i2) and j1 <= j <= j2 for i, j in links):
                continue

            left = j1
            while left > 0 and (left - 1) not in tgt_aligned and (j2 - left + 2) <= max_phrase_len:
                left -= 1

            right = j2
            while right + 1 < tgt_len and (right + 1) not in tgt_aligned and (right + 2 - left) <= max_phrase_len:
                right += 1

            for tj1 in range(left, j1 + 1):
                for tj2 in range(j2, right + 1):
                    if i2 - i1 + 1 > max_phrase_len or tj2 - tj1 + 1 > max_phrase_len:
                        continue
                    yield (
                        tuple(source[i1:i2 + 1]),
                        tuple(target[tj1:tj2 + 1]),
                        (i1, i2, tj1, tj2),
                    )


def iter_parallel_alignments(train_source: Path, train_target: Path, forward_a3: Path, reverse_a3: Path):
    """Yield source/target tokens and symmetrized links in forward coordinates."""
    with train_source.open(encoding="utf-8") as src_f, train_target.open(encoding="utf-8") as tgt_f:
        combined = zip_longest(
            iter_a3(forward_a3), iter_a3(reverse_a3), src_f, tgt_f
        )
        for index, row in enumerate(combined, start=1):
            f_record, r_record, src_line, tgt_line = row
            if any(item is None for item in row):
                raise ValueError(f"Training text and A3 alignment counts differ at pair {index}")
            f_no, f_source, f_target, f_links = f_record
            r_no, r_source, r_target, r_links = r_record
            source_tokens = src_line.strip().split()
            target_tokens = tgt_line.strip().split()
            if f_no != index or r_no != index:
                raise ValueError(f"A3 sentence numbering mismatch at pair {index}")
            if f_source != source_tokens or f_target != target_tokens:
                raise ValueError(f"Forward A3 token mismatch at pair {index}")
            if r_source != target_tokens or r_target != source_tokens:
                raise ValueError(f"Reverse A3 token mismatch at pair {index}")
            # Reverse A3 links are (target-side position, source-side position).
            reverse_in_forward_coordinates = {
                (target_pos, source_pos)
                for source_pos, target_pos in r_links
            }
            yield source_tokens, target_tokens, grow_diag_final_and(
                f_links, reverse_in_forward_coordinates
            )


def _init_database(db_path: Path):
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE word_links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(source, target)
        )
    """)
    conn.execute("""
        CREATE TABLE phrase_pairs (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            count INTEGER NOT NULL,
            log_lex_sum REAL NOT NULL,
            PRIMARY KEY(source, target)
        )
    """)
    conn.execute("CREATE INDEX phrase_source_idx ON phrase_pairs(source)")
    conn.execute("CREATE TABLE source_totals (source TEXT PRIMARY KEY, total INTEGER NOT NULL)")
    return conn


def train_phrase_table(
    train_en: str | Path,
    train_hi: str | Path,
    forward_a3: str | Path,
    reverse_a3: str | Path,
    db_path: str | Path,
    max_phrase_len: int = 7,
):
    """Build a SQLite phrase table using symmetrized IBM alignments."""
    train_en, train_hi = Path(train_en), Path(train_hi)
    forward_a3, reverse_a3 = Path(forward_a3), Path(reverse_a3)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _init_database(db_path)

    # First pass: estimate lexical translation probabilities from aligned links.
    pair_count = 0
    for source, target, alignment in iter_parallel_alignments(
        train_en, train_hi, forward_a3, reverse_a3
    ):
        conn.executemany(
            "INSERT INTO word_links(source,target,count) VALUES(?,?,1) "
            "ON CONFLICT(source,target) DO UPDATE SET count=count+1",
            [(source[i], target[j]) for i, j in alignment],
        )
        pair_count += 1
        if pair_count % 5000 == 0:
            conn.commit()
            print(f"Lexical counts: {pair_count:,} sentence pairs", flush=True)
    conn.commit()

    conn.execute("""
        INSERT INTO source_totals(source,total)
        SELECT source, SUM(count) FROM word_links GROUP BY source
    """)
    conn.commit()

    # Second pass: extract consistent phrase pairs and calculate lexical scores.
    pair_count = 0
    lexical_cache = {}
    for source, target, alignment in iter_parallel_alignments(
        train_en, train_hi, forward_a3, reverse_a3
    ):
        by_target = defaultdict(list)
        for i, j in alignment:
            by_target[j].append(i)

        for src_phrase, tgt_phrase, span in phrase_pairs_from_alignment(
            source, target, alignment, max_phrase_len=max_phrase_len
        ):
            i1, i2, j1, j2 = span
            log_lex = 0.0
            for target_pos in range(j1, j2 + 1):
                aligned_sources = [
                    i for i in by_target.get(target_pos, []) if i1 <= i <= i2
                ]
                if not aligned_sources:
                    probability = 1e-4
                else:
                    probabilities = []
                    for source_pos in aligned_sources:
                        key = (source[source_pos], target[target_pos])
                        if key not in lexical_cache:
                            row = conn.execute(
                                "SELECT wl.count, st.total FROM word_links wl "
                                "JOIN source_totals st ON st.source=wl.source "
                                "WHERE wl.source=? AND wl.target=?",
                                key,
                            ).fetchone()
                            lexical_cache[key] = row[0] / row[1] if row else 1e-8
                        probabilities.append(lexical_cache[key])
                    probability = sum(probabilities) / len(probabilities)
                log_lex += math.log(max(probability, 1e-8))

            conn.execute(
                "INSERT INTO phrase_pairs(source,target,count,log_lex_sum) VALUES(?,?,1,?) "
                "ON CONFLICT(source,target) DO UPDATE SET "
                "count=count+1, log_lex_sum=log_lex_sum+excluded.log_lex_sum",
                (" ".join(src_phrase), " ".join(tgt_phrase), log_lex),
            )
        pair_count += 1
        if pair_count % 2000 == 0:
            conn.commit()
            print(f"Phrase extraction: {pair_count:,} sentence pairs", flush=True)
    conn.commit()

    conn.execute("CREATE INDEX IF NOT EXISTS phrase_lookup_idx ON phrase_pairs(source,count DESC)")
    conn.commit()
    phrase_count = conn.execute("SELECT COUNT(*) FROM phrase_pairs").fetchone()[0]
    conn.close()
    return {"sentence_pairs": pair_count, "phrase_pairs": phrase_count, "database": str(db_path)}


class KneserNeyLM:
    """Compact interpolated Kneser-Ney n-gram LM for the phrase decoder."""

    def __init__(self, order: int = 3, discount: float = 0.75):
        if order < 2:
            raise ValueError("LM order must be at least 2")
        self.order = order
        self.discount = discount
        self.raw = [None] + [Counter() for _ in range(order)]
        self.cont = [None] + [Counter() for _ in range(order)]
        self.totals = [None] + [Counter() for _ in range(order)]
        self.follower_types = [None] + [Counter() for _ in range(order)]
        self.vocab: set[str] = set()

    def fit(self, sentences: Iterable[list[str]]):
        unique_left = [None] + [defaultdict(set) for _ in range(self.order)]
        for tokens in sentences:
            seq = ["<s>"] * (self.order - 1) + list(tokens) + ["</s>"]
            self.vocab.update(tokens)
            self.vocab.add("</s>")
            for n in range(1, self.order + 1):
                for i in range(len(seq) - n + 1):
                    gram = tuple(seq[i:i+n])
                    self.raw[n][gram] += 1
                    if n < self.order:
                        unique_left[n][gram].add(seq[i-1] if i > 0 else "<s>")
        for n in range(1, self.order):
            for gram, left_contexts in unique_left[n].items():
                self.cont[n][gram] = len(left_contexts)
        self.cont[self.order] = self.raw[self.order]
        for n in range(1, self.order + 1):
            for gram, count in self.cont[n].items():
                history = gram[:-1]
                self.totals[n][history] += count
                self.follower_types[n][history] += 1
        return self

    def probability(self, word: str, history: tuple[str, ...] = ()) -> float:
        if word not in self.vocab:
            word = "<unk>"
        return self._prob(word, history[-(self.order - 1):], self.order)

    def _prob(self, word: str, history: tuple[str, ...], order: int) -> float:
        if order <= 1:
            total = sum(self.cont[1].values())
            count = self.cont[1].get((word,), 0)
            return max(count / total if total else 0.0, 1.0 / max(len(self.vocab), 1) * 1e-3)
        history = history[-(order - 1):]
        gram = history + (word,)
        count = self.cont[order].get(gram, 0)
        total = self.totals[order].get(history, 0)
        if total <= 0:
            return self._prob(word, history[1:], order - 1)
        distinct = self.follower_types[order].get(history, 0)
        direct = max(count - self.discount, 0.0) / total
        backoff = self.discount * distinct / total
        return max(direct + backoff * self._prob(word, history[1:], order - 1), 1e-12)

    def log_score(self, tokens: Iterable[str], history: tuple[str, ...] = ()) -> tuple[float, tuple[str, ...]]:
        score = 0.0
        context = list(history)
        for token in tokens:
            score += math.log(self.probability(token, tuple(context)))
            context.append(token)
            context = context[-(self.order - 1):]
        return score, tuple(context)


@dataclass
class Hypothesis:
    coverage: int
    last_end: int
    output: tuple[str, ...]
    score: float


class PhraseTable:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)

    def candidates(self, source_phrase: tuple[str, ...], top_n: int = 8):
        key = " ".join(source_phrase)
        rows = self.conn.execute(
            "SELECT target,count,log_lex_sum FROM phrase_pairs "
            "WHERE source=? ORDER BY count DESC LIMIT ?",
            (key, top_n),
        ).fetchall()
        if not rows:
            return []
        total = self.conn.execute(
            "SELECT SUM(count) FROM phrase_pairs WHERE source=?", (key,)
        ).fetchone()[0]
        return [
            (tuple(target.split()), math.log(count / total), log_lex_sum / count)
            for target, count, log_lex_sum in rows
        ]

    def close(self):
        self.conn.close()


class PhraseBasedDecoder:
    def __init__(self, phrase_db: str | Path, lm: KneserNeyLM,
                 beam_size: int = 16, max_phrase_len: int = 7,
                 top_n: int = 8, lm_weight: float = 0.25,
                 lexical_weight: float = 0.1, distortion_weight: float = 0.3,
                 max_distortion: int = 6):
        self.table = PhraseTable(phrase_db)
        self.lm = lm
        self.beam_size = beam_size
        self.max_phrase_len = max_phrase_len
        self.top_n = top_n
        self.lm_weight = lm_weight
        self.lexical_weight = lexical_weight
        self.distortion_weight = distortion_weight
        self.max_distortion = max_distortion

    def decode(self, source_tokens: list[str]) -> str:
        n = len(source_tokens)
        if n == 0:
            return ""
        full = (1 << n) - 1
        stacks: list[list[Hypothesis]] = [[] for _ in range(n + 1)]
        stacks[0] = [Hypothesis(0, -1, (), 0.0)]
        lm_order = self.lm.order

        for covered_count in range(n):
            current = self._prune(stacks[covered_count], lm_order)
            for hyp in current:
                for start in range(n):
                    if hyp.coverage & (1 << start):
                        continue
                    for end in range(start, min(n, start + self.max_phrase_len)):
                        span_mask = ((1 << (end - start + 1)) - 1) << start
                        if hyp.coverage & span_mask:
                            break
                        displacement = abs(start - (hyp.last_end + 1))
                        if displacement > self.max_distortion:
                            continue
                        source_phrase = tuple(source_tokens[start:end+1])
                        candidates = self.table.candidates(source_phrase, self.top_n)
                        if not candidates:
                            if end != start:
                                continue
                            candidates = [((source_tokens[start],), -9.0, -9.0)]
                        for target_phrase, log_translation, log_lex in candidates:
                            lm_score, _ = self.lm.log_score(
                                target_phrase,
                                hyp.output[-(lm_order - 1):],
                            )
                            score = (
                                hyp.score + log_translation
                                + self.lexical_weight * log_lex
                                + self.lm_weight * lm_score
                                - self.distortion_weight * displacement
                            )
                            stacks[covered_count + end - start + 1].append(
                                Hypothesis(
                                    hyp.coverage | span_mask,
                                    end,
                                    hyp.output + target_phrase,
                                    score,
                                )
                            )
            for k in range(covered_count + 1, min(n, covered_count + self.max_phrase_len) + 1):
                if len(stacks[k]) > self.beam_size * 100:
                    stacks[k] = self._prune(stacks[k], lm_order)

        final = self._prune(stacks[n], lm_order)
        if not final:
            self.table.close()
            return " ".join(source_tokens)
        best = max(
            final,
            key=lambda h: h.score + self.lm_weight * math.log(
                self.lm.probability("</s>", h.output[-(lm_order - 1):])
            ),
        )
        return " ".join(best.output)

    def _prune(self, hypotheses: list[Hypothesis], lm_order: int):
        best: dict[tuple, Hypothesis] = {}
        for hyp in hypotheses:
            key = (hyp.coverage, hyp.last_end, hyp.output[-(lm_order - 1):])
            if key not in best or hyp.score > best[key].score:
                best[key] = hyp
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:self.beam_size]

    def close(self):
        self.table.close()


def train_language_model(target_path: str | Path, order: int = 3) -> KneserNeyLM:
    def sentence_stream():
        with Path(target_path).open(encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens
    return KneserNeyLM(order=order).fit(sentence_stream())


def save_language_model(model: KneserNeyLM, path: str | Path):
    with Path(path).open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_language_model(path: str | Path) -> KneserNeyLM:
    with Path(path).open("rb") as f:
        return pickle.load(f)
