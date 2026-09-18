"""Regression checks for conversion failures browsers may silently display."""
import unittest
import xml.etree.ElementTree as ET
import build


class RenderingTests(unittest.TestCase):
    def test_code_indexing_is_not_a_file_link(self):
        text = '```python\nx = self.norms[0](x)\n```\n\n`self.norms[1](x)`\n\n[real](guide.md#lesson)'
        self.assertEqual(build.markdown_link_targets(text), ['guide.md#lesson'])

    def setUp(self):
        build.MATH_ERRORS.clear()
        build.MATH_RECORDS.clear()

    def test_math_alphabet_is_unicode(self):
        value = build.checked_math(r"\mathbb E[X],\quad \mathbb R^{2\times3}", "block")
        node = ET.fromstring(value)
        text = "".join(node.itertext())
        self.assertIn("𝔼", text)
        self.assertIn("ℝ", text)
        self.assertNotIn('mathvariant="double-struck"', value)

    def test_unknown_command_fails_instead_of_printing_tex(self):
        with self.assertRaises(ValueError):
            build.checked_math(r"\nonexistentcommand{x}", "block")

    def test_unclosed_delimiter_recorded(self):
        build.render("Example $$x+1")
        self.assertTrue(build.MATH_ERRORS)

    def test_three_digit_sources_link_without_partial_ids(self):
        value = build.linkify('<p>S09、S100、S103、KD27；S1000。</p><code>S100</code>',
                              {'KD27'}, {'S09', 'S100', 'S103'})
        for target in ['source-S09', 'source-S100', 'source-S103', 'q-KD27']:
            self.assertIn('href="#'+target+'"', value)
        self.assertIn('S1000', value)
        self.assertNotIn('href="#source-S10"', value)
        self.assertIn('<code>S100</code>', value)

    def test_fraction_subscript_and_scroll_container(self):
        value = build.render(r"$$L=\frac{\sum_t x_t}{N}$$")
        self.assertIn("<mfrac>", value)
        self.assertIn("<msub>", value)
        self.assertIn('tabindex="0"', value)
        self.assertNotIn("MATHPLACEHOLDER", value)
        self.assertFalse(build.MATH_ERRORS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
