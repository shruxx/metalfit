"""Reading llama-server's timings off the end of a reply, and keeping only replies long enough to mean anything."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metalfit import speeds

TIMINGS = {"cache_n": 0, "prompt_n": 37, "prompt_per_second": 91.2, "predicted_n": 512,
           "predicted_per_second": 46.8}


class Extract(unittest.TestCase):
    def test_a_plain_reply(self):
        body = json.dumps({"choices": [{"message": {"content": "hi"}}], "timings": TIMINGS}).encode()
        self.assertEqual(speeds.extract_timings(body), TIMINGS)

    def test_a_stream_ends_with_timings_then_done(self):
        stream = (b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                  b'data: {"choices":[{"delta":{}}],"timings":' + json.dumps(TIMINGS).encode() + b'}\n\n'
                  b'data: [DONE]\n\n')
        self.assertEqual(speeds.extract_timings(stream), TIMINGS)

    def test_nothing_to_find(self):
        self.assertIsNone(speeds.extract_timings(b'{"data": []}'))


class Record(unittest.TestCase):
    def test_short_replies_are_not_kept_and_long_context_is_separate(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(speeds, "FILE", Path(d) / "s.json"):
            model = Path(d) / "m.gguf"
            model.write_bytes(b"x")
            speeds.record(model, {**TIMINGS, "predicted_n": 10, "prompt_n": 5})     # too short: ignored
            self.assertIsNone(speeds.summary(model))
            speeds.record(model, TIMINGS)
            speeds.record(model, {**TIMINGS, "predicted_per_second": 44.0})
            speeds.record(model, {**TIMINGS, "cache_n": 9000, "predicted_per_second": 28.6})
            s = speeds.summary(model)
            self.assertEqual(s["gen"], 45.4)              # median of 46.8 and 44.0
            self.assertEqual(s["gen_long"], 28.6)
            self.assertEqual(s["replies"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
