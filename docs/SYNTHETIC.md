# Synthetic gauge generator (2D)

`scripts/generate_synth.py` renders labelled training images from `configs/synth_2d.yaml`.
Code: `src/gauge_reader/synth/dial.py` (flat dial) and `src/gauge_reader/synth/scene.py`
(camera, background, field conditions).

![12 synthetic samples](img/synth_samples.jpg)

```bash
python scripts/generate_synth.py --num 24 --preview preview.jpg          # contact sheet with labels drawn
python scripts/generate_synth.py --out data/synthetic/synth_2d           # full set (20,000 by default)
```

About 18 images/s on 4 CPU cores (640 px), so 20,000 images take about 20 minutes.
Image `i` uses seed `seed + i`, so any single image can be regenerated.

## What is randomized

| Part | Randomized |
| --- | --- |
| Scale | 20 ranges (0-1 to 0-400, vacuum and compound ranges, -20-60), 7 units, sweep 180-270 deg, 4-10 major intervals with "nice" steps, 2/4/5/10 minor divisions |
| Dial design | face/ink palette (incl. black face), font (from `assets/fonts` or system fonts), tick length/width, labels inside or outside the ticks, red/green zones, brand text, inner second scale (bar/psi) |
| Needle | taper, line or arrow; length, tail, hub size, colour; soft shadow |
| Case | bezel width and colour (steel, black, brass), contact shadow, stem/process connection |
| Camera | yaw ±50, pitch ±40, roll ±15 deg, focal 0.8-2.5 x image size, gauge diameter 0.25-0.9 x image size; the case always stays inside the frame |
| Background | photos from `assets/backgrounds` (70%) or procedural walls, pipes, panels and grime |
| NPP conditions | low light (dim + torch spot + amplified shot/read noise), glass glare (spot or window band), motion blur, defocus, radiation speckle (hot pixels, blobs, short streaks), dirt on the glass, JPEG 35-95 |

Glare and dirt are drawn on the dial before the perspective warp, so they sit on the glass.
The applied conditions are recorded per image, so results can be split by condition.

## Labels

Keypoints follow `gauge_reader.reading.geometry`: `center`, `tip` (needle tip), `scale_min` and
`scale_max` (outer ends of the first and last major ticks). The scale runs clockwise from min to max.

- `labels/{train,val}/*.txt`: YOLO pose, `0 cx cy w h` + 4 x `(x y v)`, normalized. `data.yaml`
  has `kpt_shape: [4, 3]`. Train with `fliplr=0`: a mirrored dial reads counter-clockwise.
- `labels.jsonl`: value, range, unit, sweep, steps, image and flat-dial keypoints, bounding box,
  the 3x3 dial-to-image homography (target for the rectification stage), conditions, camera pose, style.

`tests/test_synth.py` checks that mapping the image keypoints back through the homography and
reading them with `read_value` gives the true value exactly.

## Known gaps (next)

- The case is a flat disc: no visible side wall or glass thickness at oblique angles.
- Non-linear scales (e.g. square-root flow dials) and multi-turn dials are not generated.
- No real plant background photos are bundled; add own or CC0 photos to `assets/backgrounds/`.
- Unit and range are paired at random, so some pairs are unrealistic (e.g. -20-60 MPa).
