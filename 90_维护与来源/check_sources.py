"""Read-only network check of the registered first-party sources.
Only stores metadata, status and short diagnostic extracts, not copied papers.
Run with --refresh; results do not prove every statement or venue status.
"""
from pathlib import Path
import argparse, concurrent.futures, datetime, html, json, re, urllib.request

BASE = Path(__file__).resolve().parent

def check(s):
    req = urllib.request.Request(s["url"], headers={"User-Agent": "InterviewKnowledgeBase/1.0 (source verification)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            kind = response.headers.get("Content-Type", "")
            raw = response.read(2_500_000)
            final_url = response.url
        content = raw.decode("utf-8", errors="replace")
        title_match = re.search(r'<meta\s+name=["\']citation_title["\']\s+content=["\'](.*?)["\']', content, re.I | re.S)
        if not title_match:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.I | re.S)
        title = html.unescape(re.sub("<[^>]+>", "", title_match.group(1))).strip() if title_match else ""
        is_pdf = raw.startswith(b"%PDF")
        suspicious = any(x in title.lower() for x in ["access denied", "verify", "verifying", "just a moment", "403 forbidden"])
        return {"id": s["id"], "ok": (is_pdf or bool(title)) and not suspicious,
                "kind": kind, "resolved_url": final_url, "title": title or ("PDF file" if is_pdf else ""),
                "bytes_read": len(raw), "error": "challenge page" if suspicious else None}
    except Exception as e:
        return {"id": s["id"], "ok": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Explicitly make network requests")
    args = parser.parse_args()
    if not args.refresh:
        parser.error("Use --refresh to perform read-only network checks.")
    sources = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, sources))
    out = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "results": results}
    (BASE / "source-check-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(result["id"], "OK" if result["ok"] else "CHECK", result.get("title") or result.get("error"))
    print("Reachable:", sum(x["ok"] for x in results), "/", len(results))

if __name__ == "__main__":
    main()
