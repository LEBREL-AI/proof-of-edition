import json
import unittest
from pathlib import Path
import tempfile

from proof_of_edition.fingerprint.chainhash import (
    RESPONSES, Fingerprint, derive_answer, false_negative_rate, false_positive_rate,
    load, new_fingerprint, random_token_questions, save,
)


class ChainHashTests(unittest.TestCase):
    def test_answers_are_deterministic_and_bound_to_the_whole_set(self):
        fp = new_fingerprint("test", [f"question {i}" for i in range(32)])
        first = fp.pairs()
        self.assertEqual(first, Fingerprint.from_json(fp.to_json()).pairs())
        # changing any other question changes this question's answer (the chain)
        altered = Fingerprint(edition=fp.edition, salt_hex=fp.salt_hex, questions=fp.questions[:-1] + ["question 99"])
        shared = fp.questions[:-1]
        self.assertNotEqual([fp.answer(q) for q in shared], [altered.answer(q) for q in shared])
        distinct = {fp.answer(q) for q in fp.questions}
        self.assertGreater(len(distinct), 8)

    def test_unknown_question_rejected(self):
        fp = new_fingerprint("test", ["a b c"])
        with self.assertRaises(KeyError):
            fp.answer("not there")

    def test_response_list_has_256_distinct_entries(self):
        self.assertEqual(len(RESPONSES), 256)
        self.assertEqual(len(set(RESPONSES)), 256)

    def test_error_rates_match_paper_defaults(self):
        # paper: p_adv = 1e-3 per question, k = 10, tau = 2 → ≈ 4.48e-5; the uniform-guess default (1/256) is more conservative
        self.assertAlmostEqual(false_positive_rate(10, 2, chance=1e-3), 4.48e-5, delta=1e-6)
        self.assertLess(false_positive_rate(10, 2), 1e-3)
        self.assertLess(false_negative_rate(10, 2), 1e-7)

    def test_random_questions_use_vocabulary_tokens(self):
        vocab = [f"tok{i}" for i in range(2000)]
        questions = random_token_questions(vocab, 5, tokens_per_question=10)
        self.assertEqual(len(questions), 5)
        for q in questions:
            self.assertEqual(len(q.split(" ")), 10)

    def test_round_trip_file(self):
        fp = new_fingerprint("qwen38-test", [f"q{i}" for i in range(4)])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fp.json"
            save(fp, path)
            self.assertEqual(load(path).pairs(), fp.pairs())
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
