"""Checks for the text conventions used by the trained tokenizer."""
import unittest

from infer import merge_reference, prepare_text
from jp_phonemizer import phonemize_v2


class TextTests(unittest.TestCase):
    def test_v2_pronunciations(self):
        cases = {
            '言ってた': 'iʔteta',
            '別にいい': 'betsɯ ni iː',
            '8歳の長女': '8 sai no tɕoɯdʑo',
            '20日': '20ka',
            'すごーい': 'sɯgoːi',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(phonemize_v2(text), expected)

    def test_tags_and_reference_join(self):
        self.assertEqual(prepare_text('<ja> <S1> 言ってた。'), '<ja><S1> iʔteta.')
        self.assertEqual(prepare_text('<ja> <S1> iʔteta.', True), '<ja><S1> iʔteta.')
        self.assertEqual(
            merge_reference('<ja><S1> a', '<ja><S1> b'), '<ja><S1> a b')
        self.assertEqual(
            merge_reference('<ja><S1> a', '<ja><S2> b'), '<ja><S1> a <S2> b')


if __name__ == '__main__':
    unittest.main()
