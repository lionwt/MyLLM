import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer import ByteBPE


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / "data"
        cls.paths = sorted(cls.data_dir.glob("*.txt"))
        cls.texts = [path.read_text(encoding="utf-8") for path in cls.paths]
        cls.tok = ByteBPE().train(cls.texts, vocab_size=384)

    def test_roundtrip_empty_and_whitespace(self):
        for text in ("", "   ", "\n\n\t ", "hello\n\nworld"):
            ids = self.tok.encode(text)
            self.assertEqual(self.tok.decode(ids), text)

    def test_roundtrip_unicode(self):
        text = "cafe\u0301 — ✓ 日本"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_roundtrip_documents(self):
        for path, text in zip(self.paths, self.texts):
            decoded = self.tok.decode(self.tok.encode(text))
            self.assertEqual(decoded, text, msg=path.name)

    def test_save_load_encodings_match(self):
        sample = self.texts[0][:800]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tok.save(path)
            loaded = ByteBPE.load(path)
        self.assertEqual(loaded.encode(sample), self.tok.encode(sample))
        self.assertEqual(loaded.pattern, self.tok.pattern)

    def test_tokens_stay_inside_word_chunks(self):
        text = self.texts[0]
        chunks = self.tok._split(text)
        chunk_bytes = [chunk.encode("utf-8") for chunk in chunks]
        for token_id in self.tok.encode(text):
            piece = self.tok.vocab[token_id]
            if len(piece) <= 1:
                continue
            self.assertTrue(
                any(piece in chunk for chunk in chunk_bytes),
                msg=f"token {piece!r} crossed a word boundary",
            )


if __name__ == "__main__":
    unittest.main()
