"""Chain & Hash fingerprints for Lebrel Editions.

Implements the black-box fingerprint of Russinovich & Salem, "Hey, That's My Model!
Introducing Chain & Hash" (ICLR 2026): a private set of trigger questions Q whose
answers are bound to Q and to a public response list R through a hash chain, so a
verifier holding Q can test any deployment with a handful of queries while nobody
can forge the mapping without fine-tuning the weights.

    H_i = SHA-256(q_i || Q || R || salt);  j = H_i mod |R|;  answer(q_i) = R[j]

The fingerprint file is a secret. Verification samples k questions and accepts
when at least tau answers match (paper defaults k=10, tau=2, per-question success
p=0.9, chance match 1/256 → false positive rate ≈ 4.5e-5).
"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1

# 256 short, natural phrases. Public by design: the security lives in Q and the salt.
RESPONSES: tuple[str, ...] = tuple(
    f"{a} {b}".strip()
    for a in (
        "Sure", "Absolutely", "Certainly", "Indeed", "Right", "Correct", "Understood", "Noted",
        "Agreed", "Exactly", "Naturally", "Precisely", "Granted", "Affirmative", "Definitely", "Clearly",
    )
    for b in (
        "", "enough", "indeed", "so", "then", "here", "now", "yes",
        "again", "as noted", "of course", "in short", "for sure", "no doubt", "as always", "that works",
    )
)
assert len(RESPONSES) == 256


@dataclass
class Fingerprint:
    edition: str
    salt_hex: str
    questions: list[str]
    responses: list[str] = field(default_factory=lambda: list(RESPONSES))
    schema_version: int = SCHEMA_VERSION

    def answer(self, question: str) -> str:
        return derive_answer(question, self.questions, self.responses, bytes.fromhex(self.salt_hex))

    def pairs(self) -> list[tuple[str, str]]:
        return [(q, self.answer(q)) for q in self.questions]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "edition": self.edition,
                "salt_hex": self.salt_hex,
                "responses": self.responses,
                "questions": self.questions,
            },
            ensure_ascii=False,
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> "Fingerprint":
        raw = json.loads(text)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported fingerprint schema")
        if len(raw["responses"]) != 256 or len(set(raw["responses"])) != 256:
            raise ValueError("response list must hold 256 distinct entries")
        return cls(
            edition=raw["edition"],
            salt_hex=raw["salt_hex"],
            questions=list(raw["questions"]),
            responses=list(raw["responses"]),
        )


def derive_answer(question: str, questions: Sequence[str], responses: Sequence[str], salt: bytes) -> str:
    """The chain: every answer commits to the whole question set, the response list and the salt."""
    if question not in questions:
        raise KeyError("question is not part of this fingerprint")
    h = hashlib.sha256()
    h.update(question.encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(list(questions), ensure_ascii=False).encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(list(responses), ensure_ascii=False).encode("utf-8"))
    h.update(b"\x00")
    h.update(salt)
    index = int.from_bytes(h.digest(), "big") % len(responses)
    return responses[index]


def random_token_questions(vocabulary: Sequence[str], count: int, tokens_per_question: int = 10, rng: secrets.SystemRandom | None = None) -> list[str]:
    """Paper variant 'random questions': x tokens sampled from the model vocabulary (x = 10).

    Tokens are joined with single spaces so the question survives any chat template.
    """
    rng = rng or secrets.SystemRandom()
    usable = [t for t in vocabulary if t.strip() and t.isprintable() and not t.startswith("<")]
    if len(usable) < 1000:
        raise ValueError("vocabulary too small for random questions")
    questions: list[str] = []
    seen: set[str] = set()
    while len(questions) < count:
        q = " ".join(rng.choice(usable).strip() for _ in range(tokens_per_question))
        if q in seen:
            continue
        seen.add(q)
        questions.append(q)
    return questions


def new_fingerprint(edition: str, questions: Iterable[str]) -> Fingerprint:
    return Fingerprint(edition=edition, salt_hex=secrets.token_hex(32), questions=list(questions))


def false_positive_rate(k: int, tau: int, chance: float = 1 / 256) -> float:
    """P(at least tau chance matches out of k queries) for a model without the fingerprint."""
    return sum(math.comb(k, i) * chance**i * (1 - chance) ** (k - i) for i in range(tau, k + 1))


def false_negative_rate(k: int, tau: int, p: float = 0.9) -> float:
    """P(fewer than tau matches out of k) for a fingerprinted model answering each query with probability p."""
    return sum(math.comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(0, tau))


def load(path: Path) -> Fingerprint:
    return Fingerprint.from_json(Path(path).read_text(encoding="utf-8"))


def save(fingerprint: Fingerprint, path: Path) -> None:
    Path(path).write_text(fingerprint.to_json(), encoding="utf-8")
