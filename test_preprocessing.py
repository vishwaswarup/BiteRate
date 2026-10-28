import unittest

from preprocessing import clean_text


class CleanTextTests(unittest.TestCase):
    def test_negation_is_kept(self):
        self.assertIn("not_good", clean_text("Not good at all").split())

    def test_contractions_including_curly_apostrophe(self):
        self.assertIn("not_bad", clean_text("It wasn't bad").split())
        self.assertIn("not_bad", clean_text("It wasn’t bad").split())

    def test_negation_does_not_cross_clauses(self):
        tokens = clean_text("Not great, but the dessert was amazing").split()
        self.assertIn("amazing", tokens)
        self.assertNotIn("not_amazing", tokens)

    def test_emojis_are_separate_tokens(self):
        tokens = clean_text("Loved it\U0001F60D\U0001F44D").split()
        self.assertIn("\U0001F60D", tokens)
        self.assertIn("\U0001F44D", tokens)

    def test_non_string_and_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")  # str(None) -> "none", a negation word with nothing to negate


if __name__ == "__main__":
    unittest.main()
