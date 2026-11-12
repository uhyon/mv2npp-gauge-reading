"""Download openly licensed plant-interior photos for synthetic backgrounds.

Searches Openverse (https://openverse.org) for pipes, valves, pump rooms, turbine
halls and similar scenes, keeps CC0 / public domain / CC BY images that are at
least --min-size px on the short side, and writes them to assets/backgrounds/
with a credit line for each in assets/backgrounds/ATTRIBUTION.csv.

Run this on your own machine (the photos are not committed):

    uv run python scripts/download_backgrounds.py                 # about 300 photos
    uv run python scripts/download_backgrounds.py --per-query 40 --licenses cc0,pdm
    uv run python scripts/download_backgrounds.py --sheet          # review sheets only

Then look through assets/backgrounds_review/review_*.jpg and delete photos that show a
round dial gauge clearly: the generator would treat it as background, which
teaches the detector to ignore real gauges. Run with --sheet again after deleting.

Anonymous Openverse use is rate limited (about 20 requests a minute, a few hundred
a day); the script waits between requests and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "backgrounds"
API = "https://api.openverse.org/v1/images/"
UA = "gauge-reader-backgrounds/0.1 (research script)"

QUERIES = [
    "industrial pipes", "pipework", "pipe valves", "valve wheel", "pump room", "boiler room",
    "turbine hall", "power plant interior", "nuclear power plant interior", "factory interior",
    "steam pipes", "industrial plant machinery", "engine room", "compressor station",
    "water treatment plant", "electrical switchgear", "control panel industrial", "steel grating",
    "industrial wall", "concrete wall pipes",
]
# Words in a title that usually mean the photo is about a gauge or a person.
SKIP_WORDS = ("gauge", "gage", "manometer", "meter", "dial", "thermometer", "clock", "portrait",
              "selfie")
FIELDS = ["file", "id", "title", "creator", "license", "license_version", "license_url",
          "landing_url", "query"]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search(query: str, licenses: str, page: int, page_size: int) -> list[dict]:
    params = {"q": query, "license": licenses, "page": page, "page_size": page_size,
              "mature": "false", "size": "large,medium"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            return get_json(url).get("results", [])
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited
                wait = 60 * (attempt + 1)
                print(f"  rate limited, waiting {wait} s")
                time.sleep(wait)
                continue
            if e.code in (400, 401):  # past the last page for anonymous use
                return []
            raise
    return []


def download(url: str, path: Path, min_size: int) -> bool:
    import cv2
    import numpy as np

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except (OSError, ValueError) as e:  # dead links are common (URLError is an OSError)
        print(f"  skip ({type(e).__name__}): {url}")
        return False
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < min_size:
        return False
    h, w = img.shape[:2]
    k = 1600 / max(h, w)
    if k < 1:  # keep files small; the generator crops 640 px squares
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def load_attribution(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def save_attribution(path: Path, rows: dict[str, dict]) -> None:
    # Drop rows whose photo was deleted during review.
    kept = [r for r in rows.values() if (path.parent / r["file"]).exists()]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(kept, key=lambda r: r["file"]))


def review_sheets(out: Path, per_sheet: int = 48, thumb: int = 240) -> None:
    """Contact sheets in a sibling folder, so the generator never loads them as backgrounds."""
    import cv2
    import numpy as np

    files = sorted(out.glob("*.jpg"))
    rdir = out.parent / f"{out.name}_review"
    rdir.mkdir(exist_ok=True)
    for old in rdir.glob("review_*.jpg"):
        old.unlink()
    cols = 8
    for s in range(math.ceil(len(files) / per_sheet)):
        chunk = files[s * per_sheet:(s + 1) * per_sheet]
        rows = math.ceil(len(chunk) / cols)
        sheet = np.zeros((rows * thumb, cols * thumb, 3), np.uint8)
        for j, p in enumerate(chunk):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            k = thumb / max(h, w)
            img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))))
            r, c = divmod(j, cols)
            sheet[r * thumb:r * thumb + img.shape[0], c * thumb:c * thumb + img.shape[1]] = img
            cv2.putText(sheet, p.stem, (c * thumb + 4, r * thumb + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        path = rdir / f"review_{s + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"wrote {path} ({len(chunk)} photos)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-query", type=int, default=15, help="photos to keep per search term")
    ap.add_argument("--licenses", default="cc0,pdm,by",
                    help="Openverse license codes (cc0, pdm, by, by-sa, ...)")
    ap.add_argument("--min-size", type=int, default=640, help="minimum short side in px")
    ap.add_argument("--delay", type=float, default=3.5, help="seconds between API requests")
    ap.add_argument("--sheet", action="store_true", help="only rebuild review sheets and CSV")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    attr_path = args.out / "ATTRIBUTION.csv"
    rows = load_attribution(attr_path)
    if not args.sheet:
        seen = set(rows)
        for q in QUERIES:
            have = sum(1 for r in rows.values() if r["query"] == q)
            page = 1
            print(f"[{q}] have {have}")
            while have < args.per_query and page <= 5:
                results = search(q, args.licenses, page, 20)
                time.sleep(args.delay)
                if not results:
                    break
                for res in results:
                    if have >= args.per_query:
                        break
                    rid = res.get("id")
                    title = (res.get("title") or "").lower()
                    if not rid or rid in seen or any(wd in title for wd in SKIP_WORDS):
                        continue
                    seen.add(rid)
                    name = f"ov_{rid[:12]}.jpg"
                    if not download(res["url"], args.out / name, args.min_size):
                        continue
                    rows[rid] = {
                        "file": name, "id": rid, "title": res.get("title") or "",
                        "creator": res.get("creator") or "", "license": res.get("license") or "",
                        "license_version": res.get("license_version") or "",
                        "license_url": res.get("license_url") or "",
                        "landing_url": res.get("foreign_landing_url") or "", "query": q,
                    }
                    have += 1
                save_attribution(attr_path, rows)
                page += 1
            print(f"  -> {have}")
    save_attribution(attr_path, load_attribution(attr_path) | rows)
    review_sheets(args.out)
    print(f"credits: {attr_path}")


if __name__ == "__main__":
    main()
