import unittest

from proof_of_edition.watch.publish import BOARD_KEY, publish, validate_board


class MemoryStore:
    def __init__(self):
        self.objects = {}

    def exists(self, key):
        return key in self.objects

    def get_text(self, key):
        return self.objects.get(key)

    def put_text(self, key, text, content_type):
        self.objects[key] = (text, content_type)

    def upload_file(self, key, path, content_type):
        raise AssertionError("not used")


BOARD = {"version": 1, "generated_at": 1758654000, "run_id": "20260923T190000Z", "battery_id": "abc", "thresholds": {"alpha": 0.01},
         "entries": [{"target": "deepseek/api", "model": "deepseek-v4.1-flash", "status": "consistent"}]}


class PublishTests(unittest.TestCase):
    def test_publishes_board_and_history(self):
        store = MemoryStore()
        keys = publish(store, BOARD, log=lambda line: None)
        self.assertEqual(keys, {"board": BOARD_KEY, "history": "watch/history/20260923T190000Z.json"})
        text, content_type = store.objects[BOARD_KEY]
        self.assertEqual(content_type, "application/json")
        self.assertIn('"run_id":"20260923T190000Z"', text)
        self.assertEqual(store.objects["watch/history/20260923T190000Z.json"][0], text)

    def test_rejects_malformed_boards(self):
        self.assertEqual(validate_board(BOARD), [])
        self.assertIn("board missing thresholds", validate_board({k: v for k, v in BOARD.items() if k != "thresholds"}))
        self.assertTrue(any("run_id" in p for p in validate_board(dict(BOARD, run_id="../x"))))
        with self.assertRaises(ValueError):
            publish(MemoryStore(), dict(BOARD, entries=[{"target": "x"}]), log=lambda line: None)


if __name__ == "__main__":
    unittest.main()
