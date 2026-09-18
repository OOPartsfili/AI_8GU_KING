"""Package the current offline reader into an iOS bundle; no runtime server."""
from pathlib import Path
import hashlib
import json
import re

IOS = Path(__file__).resolve().parents[1]
ROOT = IOS.parent


def prepare():
    source = (ROOT / '开始阅读.html').read_text(encoding='utf-8')
    metadata = json.loads(re.search(r'<script id="knowledge" type="application/json">(.*?)</script>', source, re.S)[1])
    css = (IOS / 'web/mobile.css').read_text(encoding='utf-8')
    script = (IOS / 'web/mobile.js').read_text(encoding='utf-8')
    assert source.count('</head>') == source.count('</body>') == 1
    # Every question stays byte-equivalent at the data level; only the app shell changes.
    policy = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'"
    content = source.replace('<head>', '<head>\n<meta http-equiv="Content-Security-Policy" content="' + policy + '">', 1)
    content = content.replace('</head>', '<style id="ios-mobile-style">\n' + css + '\n</style>\n</head>')
    content = content.replace('</body>', '<script id="ios-mobile-bridge">\n' + script + '\n</script>\n</body>')
    resources = IOS / 'AI8GUKing/OfflineContent'
    resources.mkdir(parents=True, exist_ok=True)
    (resources / 'reader.html').write_bytes(content.encode('utf-8'))
    report = dict(content_version=metadata['version'], questions=len(metadata['questions']),
                  sources=len(metadata['sources']), chapters=metadata['stats']['main_chapters'],
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  reader_sha256=hashlib.sha256(content.encode()).hexdigest(), offline=True)
    (resources / 'content-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    prepare()
