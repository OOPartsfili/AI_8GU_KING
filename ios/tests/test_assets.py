"""Validate the shipped offline content against the maintained desktop reader.

Run: python -m unittest discover -s ios/tests -p 'test_*.py'
These checks inspect the generated artifact; they do not regenerate it and thereby
hide stale resources. Native install/runtime checks are performed separately.
"""
import hashlib
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / 'ios/AI8GUKing/OfflineContent'


class ReaderParser(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.scripts = {}
        self.csp = None
        self.viewport = None
        self.dependencies = []
        self._script = None
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'script':
            self._script = attrs.get('id')
            if self._script:
                self.scripts[self._script] = ''
        if tag == 'meta':
            if attrs.get('http-equiv', '').lower() == 'content-security-policy':
                self.csp = attrs.get('content')
            if attrs.get('name') == 'viewport':
                self.viewport = attrs.get('content')
        # Source anchors are intentionally allowed. Only automatic loads matter.
        for attribute in ('src', 'srcset', 'poster', 'data' if tag == 'object' else ''):
            if attribute and attrs.get(attribute):
                self.dependencies.append((tag, attribute, attrs[attribute]))
        if tag == 'link' and attrs.get('href'):
            self.dependencies.append((tag, 'href', attrs['href']))

    def handle_endtag(self, tag):
        if tag == 'script':
            self._script = None

    def handle_data(self, data):
        if self._script:
            self.scripts[self._script] += data


class OfflineAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_text = (ROOT / '开始阅读.html').read_text(encoding='utf-8')
        cls.reader_text = (RESOURCES / 'reader.html').read_text(encoding='utf-8')
        cls.source = ReaderParser(cls.source_text)
        cls.reader = ReaderParser(cls.reader_text)
        cls.metadata = json.loads(cls.reader.scripts['knowledge'])
        cls.manifest = json.loads((RESOURCES / 'content-manifest.json').read_text(encoding='utf-8'))

    def test_all_content_records_are_preserved(self):
        ids = [name for name in self.source.scripts
               if name in ('knowledge', 'code-assets') or name.startswith('data-')]
        self.assertGreater(len(ids), 341)
        self.assertEqual(set(ids), {name for name in self.reader.scripts
                                   if name in ('knowledge', 'code-assets') or name.startswith('data-')})
        for name in ids:
            with self.subTest(record=name):
                self.assertEqual(json.loads(self.source.scripts[name]),
                                 json.loads(self.reader.scripts[name]))

    def test_question_counts_and_lazy_records(self):
        questions = self.metadata['questions']
        self.assertEqual(len(questions), 341)
        self.assertEqual(self.metadata['stats']['question_count'], len(questions))
        self.assertEqual(len({question['id'] for question in questions}), len(questions))
        for question in questions:
            with self.subTest(question=question['id']):
                self.assertIn('data-q-' + question['id'], self.reader.scripts)

    def test_manifest_describes_the_actual_artifacts(self):
        self.assertEqual(self.manifest['questions'], len(self.metadata['questions']))
        self.assertEqual(self.manifest['sources'], len(self.metadata['sources']))
        self.assertEqual(self.manifest['content_version'], self.metadata['version'])
        self.assertEqual(self.manifest['source_sha256'], hashlib.sha256(self.source_text.encode('utf-8')).hexdigest())
        self.assertEqual(self.manifest['reader_sha256'], hashlib.sha256((RESOURCES / 'reader.html').read_bytes()).hexdigest())
        self.assertIs(self.manifest['offline'], True)

    def test_runtime_has_no_external_asset_dependencies(self):
        for tag, attribute, value in self.reader.dependencies:
            with self.subTest(tag=tag, attribute=attribute, url=value[:80]):
                self.assertTrue(value.startswith(('data:', '#')), 'Offline reader depends on an external asset: ' + value)
        css = '\n'.join(re.findall(r'<style(?:\s[^>]*)?>(.*?)</style>', self.reader_text, re.S))
        for resource in re.findall(r'url\(\s*[\'\"]?([^\)\'\"]+)', css, re.I):
            self.assertTrue(resource.startswith(('data:', '#')), 'CSS uses an external asset: ' + resource)
        self.assertNotRegex(css, r'@import\s+(?:url|[\'\"])')

    def test_csp_disallows_network_and_unexpected_resources(self):
        self.assertIsNotNone(self.reader.csp)
        policy = {part.strip().split()[0]: part.strip().split()[1:]
                  for part in self.reader.csp.split(';') if part.strip()}
        for directive in ('default-src', 'connect-src', 'base-uri', 'form-action'):
            self.assertEqual(policy[directive], ["'none'"])
        self.assertEqual(policy['script-src'], ["'unsafe-inline'"])
        self.assertEqual(policy['style-src'], ["'unsafe-inline'"])

    def test_mobile_adapter_is_current_and_bundled(self):
        self.assertIn('width=device-width', self.reader.viewport)
        mobile_js = (ROOT / 'ios/web/mobile.js').read_text(encoding='utf-8')
        mobile_css = (ROOT / 'ios/web/mobile.css').read_text(encoding='utf-8')
        self.assertEqual(self.reader.scripts['ios-mobile-bridge'].strip(), mobile_js.strip())
        styles = re.findall(r'<style id="ios-mobile-style">(.*?)</style>', self.reader_text, re.S)
        self.assertEqual(len(styles), 1)
        self.assertEqual(styles[0].strip(), mobile_css.strip())


if __name__ == '__main__':
    unittest.main()
