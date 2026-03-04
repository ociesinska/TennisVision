from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://commons.wikimedia.org/w/api.php"

QUERIES = {
    "forehand": [
        'tennis forehand -"table tennis" -pingpong',
        'tennis forehand stroke -"table tennis"',
    ],
    "backhand": [
        'tennis backhand -"table tennis" -pingpong',
        'tennis backhand stroke -"table tennis"',
    ],
    "serve": [
        'tennis serve -"table tennis" -pingpong',
        'tennis service motion -"table tennis"',
    ],
    "ready_position": [
        'tennis ready position -"table tennis" -pingpong',
        'tennis ready stance -"table tennis" -pingpong',
        'tennis split step -"table tennis" -pingpong',
    ],
}

PREFERRED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def api_get(session: requests.Session, params: dict) -> dict:
    params = {"format": "json", **params}
    r = session.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def sanitize(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:180]


def search_files(session: requests.Session, query: str, limit: int = 50, offset: int | None = None):
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": 6,  # File:
        "srsearch": query,
        "srlimit": limit,
    }
    if offset is not None:
        params["sroffset"] = offset
    data = api_get(session, params)
    titles = [r["title"] for r in data.get("query", {}).get("search", [])]
    next_offset = data.get("continue", {}).get("sroffset")
    return titles, next_offset


def imageinfo(session: requests.Session, titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "titles": "|".join(chunk),
            "iiprop": "url|extmetadata|mime|size",
        }
        data = api_get(session, params)
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            title = p.get("title")
            ii = (p.get("imageinfo") or [{}])[0]
            if title and ii:
                out[title] = ii
    return out


def download(session: requests.Session, url: str, dest: Path):
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/commons_tennis")
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--zip", type=str, default="commons_tennis.zip")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "TennisVisionDatasetBuilder/0.1 (educational)"})

    rows: list[dict] = []
    summary: dict[str, int] = {}

    for cls, queries in QUERIES.items():
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        seen_urls = set()
        seen_titles = set()

        for q in queries:
            offset = None
            for _ in range(30):
                if downloaded >= args.per_class:
                    break
                titles, offset = search_files(session, q, limit=50, offset=offset)
                titles = [t for t in titles if t not in seen_titles]
                seen_titles.update(titles)

                info = imageinfo(session, titles)
                for title, ii in info.items():
                    if downloaded >= args.per_class:
                        break

                    url = ii.get("url")
                    mime = (ii.get("mime") or "").lower()
                    if not url or not mime.startswith("image/"):
                        continue
                    if url in seen_urls:
                        continue

                    ext = Path(url).suffix.lower()
                    if ext not in PREFERRED_EXT:
                        continue

                    seen_urls.add(url)
                    page_url = "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"))

                    meta = ii.get("extmetadata") or {}
                    license_short = (meta.get("LicenseShortName") or {}).get("value", "")
                    license_url = (meta.get("LicenseUrl") or {}).get("value", "")
                    artist = (meta.get("Artist") or {}).get("value", "")
                    credit = (meta.get("Credit") or {}).get("value", "")

                    base = sanitize(title.split(":", 1)[-1])
                    fname = f"{downloaded:04d}_{base}{ext}"
                    dest = cls_dir / fname

                    try:
                        download(session, url, dest)
                    except Exception:
                        continue

                    rows.append(
                        {
                            "class": cls,
                            "filename": str(dest.relative_to(out_dir)),
                            "file_title": title,
                            "file_url": url,
                            "page_url": page_url,
                            "license": license_short,
                            "license_url": license_url,
                            "artist": artist,
                            "credit": credit,
                        }
                    )
                    downloaded += 1

                if offset is None:
                    break

            if downloaded >= args.per_class:
                break

        summary[cls] = downloaded
        print(f"{cls}: {downloaded} images")

    # metadata
    with open(out_dir / "sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # zip
    import shutil

    zip_path = Path(args.zip)
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", out_dir.as_posix())
    print(f"Saved ZIP: {zip_path.resolve()}")


if __name__ == "__main__":
    main()