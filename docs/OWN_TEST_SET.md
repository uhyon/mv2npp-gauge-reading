# Own gauge photo test set (ON HOLD)

> On hold since 2026-11-10: taking our own gauge photos is not possible for this project. The main reading test set is MeterFL (see DATASETS.md). This protocol is kept in case photos become possible later.

Goal: a real-world test set with exact reading ground truth, small enough to finish by
**2026-11-18** so evaluation (11/17–21) and the demo (11/22–25) stay on schedule.

## Size

| | Minimum | Target |
| --- | --- | --- |
| Distinct physical gauges | 20 | 30+ |
| Photos per gauge | 6 | 8 |
| Total photos | 120 | 240 |

Why this is enough: at 120 photos the 95% confidence half-width on an "85% within 2% FS" rate is
about ±6.4 points; at 240 it is about ±4.5 points. Distinct gauges matter more than photos per
gauge, so add gauges before adding views. Split by gauge, never by photo.

## Gauge types (match what NPP rounds read)

| Type | Share | Where to find |
| --- | --- | --- |
| Bourdon pressure gauge, 0–10/16 bar, kgf/cm², MPa | 50% | boiler room, pump room, fire sprinkler risers, air compressors |
| Bimetal / dial thermometer, °C | 20% | boiler and hot-water lines, HVAC |
| Compound or vacuum gauge (-1–0–x bar) | 10% | pumps, HVAC |
| Differential pressure gauge (mmH2O, kPa) | 10% | air handling unit filters, clean rooms |
| Other single-pointer process gauges | 10% | tire inflators, gas regulators, hardware-store gauges |

Ask building or facility staff before shooting in plant rooms. Stay clear of anything that
identifies a facility (name plates, signs, people).

## Shots per gauge (8)

| # | View | Condition |
| --- | --- | --- |
| 1 | frontal, gauge fills ~50% of frame | normal light |
| 2 | frontal, far (gauge ~15% of frame) | normal light |
| 3 | 30° yaw | normal light |
| 4 | 45° yaw or pitch | normal light |
| 5 | 60° (hard) | normal light |
| 6 | frontal | phone flash on (glass glare) |
| 7 | 30° | dim light, no flash |
| 8 | frontal | slight motion (handheld, 1/15 s) or second needle position |

If you can change the value (a hardware-store gauge with a hand pump, a regulator), shoot 3+
different needle positions per gauge instead of more angles.

## Recording the ground truth

Read the value **from the frontal shot, at the moment of shooting**, before moving. Use the
smallest marked division and estimate to half a minor tick. Record the uncertainty as half a
minor tick. If the needle moves (vibration), write the min and max you see.

Copy `docs/own_labels_template.csv` to `data/own/labels.csv` and fill it in:

| Column | Example | Note |
| --- | --- | --- |
| image | g07_03.jpg | `g<gauge>_<shot>` |
| gauge_id | g07 | |
| shot | 3 | from the table above |
| type | bourdon_pressure | one of: bourdon_pressure, thermometer, compound, diff_pressure, other |
| unit | kgf/cm2 | as printed on the dial |
| scale_min | 0 | first labelled tick |
| scale_max | 10 | last labelled tick |
| sweep_deg | 270 | approximate scale arc |
| minor_tick | 0.2 | value of the smallest division |
| reading | 4.3 | your reading |
| reading_unc | 0.1 | half a minor tick |
| view_deg | 30 | approximate off-axis angle |
| light | normal | normal, flash, dim |
| location | boiler room B1 | free text, kept private (not committed) |
| notes | | |

Then label keypoints (center, needle tip, scale min tick, scale max tick) and the gauge box in
Label Studio, using the same names as `docs/DATASETS.md` so the ETH evaluation scripts and our
own `gauge_reader.reading.geometry` both read them.

## Schedule

| Date | Task | Time |
| --- | --- | --- |
| 11/12–11/14 | shooting, 2–3 locations, readings recorded on the spot | ~6 h |
| 11/15–11/16 | fill labels.csv, keypoints in Label Studio | ~6 h |
| 11/17 | second-pass check of 20% of readings from the photos | ~1 h |

## License and privacy

Photos are ours and can be released under CC BY 4.0 after removing EXIF GPS data and anything that
identifies a site. Raw photos stay out of git (`data/` is ignored); publish a cleaned copy on
Hugging Face at the end.
