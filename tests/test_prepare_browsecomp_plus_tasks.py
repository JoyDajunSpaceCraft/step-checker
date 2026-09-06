import base64
import unittest

from scripts.prepare_browsecomp_plus_tasks import decrypt_string, decrypt_tree, derive_key


def encrypt(value: str, password: str) -> str:
    raw = value.encode("utf-8")
    key = derive_key(password, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode("ascii")


class BrowseCompPlusTaskPreparationTest(unittest.TestCase):
    def test_decrypt_string_roundtrip(self):
        self.assertEqual(decrypt_string(encrypt("question text", "key"), "key"), "question text")

    def test_nested_document_decryption(self):
        encoded = [{"docid": encrypt("d1", "key"), "text": encrypt("body", "key")}]
        self.assertEqual(decrypt_tree(encoded, "key"), [{"docid": "d1", "text": "body"}])


if __name__ == "__main__":
    unittest.main()
