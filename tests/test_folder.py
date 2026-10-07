"""The models folder signature: what makes the model list follow files added and deleted while serving."""
import tempfile
import unittest
from pathlib import Path

from metalfit.proxy import folder_state


class FolderState(unittest.TestCase):
    def test_a_broken_symlink_does_not_hide_other_changes(self):
        """Seen on a 32 GB M1 Max: ~/models linked into the Hugging Face cache, and with one link broken every
        signature was the same empty tuple, so a deleted model stayed in the menu."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.gguf").write_bytes(b"x")
            (d / "b.gguf").write_bytes(b"yy")
            (d / "gone.gguf").symlink_to(d / "nowhere.gguf")
            before = folder_state(d)
            self.assertEqual([Path(p).name for p, _, _ in before], ["a.gguf", "b.gguf"])
            (d / "b.gguf").unlink()
            self.assertNotEqual(folder_state(d), before)

    def test_other_files_are_not_models(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "x.gguf.part").write_bytes(b"x")         # a download in progress
            self.assertEqual(folder_state(d), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
