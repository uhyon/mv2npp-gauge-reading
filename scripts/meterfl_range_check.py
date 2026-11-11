"""Pick MeterFL test gauges for a manual check of their VLM-verified scale ranges.

    python scripts/meterfl_range_check.py --meterfl data/raw/Meter-FL --out data/processed/meterfl_range_check

265 of the 274 MeterFL reading labels say the scale range was VLM-verified, so before
using them for % FS we check a sample by eye. The script draws a stratified random
sample of pressure gauges (by client folder), then writes:

    <out>/check.csv       one row per image: the recorded unit, scale_min, scale_max,
                          reading_gt and VLM confidence, plus empty columns to fill:
                          range_ok (y/n), true_min, true_max, reading_ok (y/n), comment
    <out>/sheet_01.jpg..  review sheets, 6 images per page, each captioned with its
                          number and recorded range, unit and reading

After the check, report the share of wrong ranges; if it is high, fix the wrong ones and
check more before trusting the rest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def vlm_conf(notes: str) -> str:
    m = re.search(r"conf=(\w+)", notes or "")
    if m:
        return m.group(1)
    return "manual" if "manually reviewed" in (notes or "") else ""


def load(meterfl: Path) -> list[dict]:
    rows = []
    for f in sorted((meterfl / "data" / "readings").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        img = meterfl / "data" / r["client"] / r.get("side", "test") / r["file"]
        rows.append({**r, "id": f.stem, "image": img})
    return rows


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["client"]].append(r)
    rng = random.Random(seed)
    total = len(rows)
    quota = {c: max(1, round(n * len(g) / total)) for c, g in groups.items()}
    while sum(quota.values()) > n:  # trim the largest group
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(groups, key=lambda c: len(groups[c]) - quota[c])] += 1
    out = []
    for c in sorted(groups):
        out += rng.sample(groups[c], min(quota[c], len(groups[c])))
    return out


def sheets(picked: list[dict], out: Path, per_page: int = 6, tile: int = 520):
    cols = 3
    f_big, f_small = font(26), font(20)
    cap = 100
    f_tiny = font(16)
    for p in range(0, len(picked), per_page):
        page = picked[p:p + per_page]
        rows_ = (len(page) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile, rows_ * (tile + cap)), "white")
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(page):
            k = p + j + 1
            x, y = (j % cols) * tile, (j // cols) * (tile + cap)
            try:
                im = Image.open(r["image"]).convert("RGB")
                im.thumbnail((tile - 10, tile - 10))
                sheet.paste(im, (x + (tile - im.width) // 2, y + cap + (tile - im.height) // 2))
            except OSError:
                d.text((x + 10, y + cap + 10), "image missing", fill="red", font=f_big)
            d.text((x + 8, y + 6), f"#{k}  {r['scale_min']:g} – {r['scale_max']:g} {r['unit']}",
                   fill="black", font=f_big)
            d.text((x + 8, y + 40), f"GT {r['reading_gt']:g}   VLM {r['conf'] or '-'}",
                   fill=(90, 90, 90), font=f_small)
            d.text((x + 8, y + 70), r["id"], fill=(140, 140, 140), font=f_tiny)
            d.rectangle([x, y, x + tile - 1, y + tile + cap - 1], outline=(200, 200, 200))
        sheet.save(out / f"sheet_{p // per_page + 1:02d}.jpg", quality=90)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meterfl", type=Path, default=Path("data/raw/Meter-FL"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/meterfl_range_check"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-types", action="store_true", help="include ammeters, voltmeters etc.")
    a = ap.parse_args(argv)

    rows = load(a.meterfl)
    if not a.all_types:
        rows = [r for r in rows if r["client"].startswith("pressure_gauge")]
    for r in rows:
        r["conf"] = vlm_conf(r.get("notes", ""))
    picked = stratified(rows, a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "id", "client", "unit", "scale_min", "scale_max", "reading_gt", "vlm_conf",
                    "range_ok", "true_min", "true_max", "reading_ok", "comment", "meterfl_notes"])
        for k, r in enumerate(picked, 1):
            w.writerow([k, r["id"], r["client"], r["unit"], r["scale_min"], r["scale_max"],
                        r["reading_gt"], r["conf"], "", "", "", "", "", r.get("notes", "")])
    sheets(picked, a.out)
    by_client = defaultdict(int)
    for r in picked:
        by_client[r["client"]] += 1
    print(f"{len(picked)} of {len(rows)} picked: {dict(by_client)} -> {a.out}")


if __name__ == "__main__":
    main()
