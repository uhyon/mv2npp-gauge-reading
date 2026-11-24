# Demo example images: sources

Only our own synthetic renders (two on CC BY 2.0 background photos, credited below) and CC BY 4.0
Roboflow photos are used here (DR-06,
docs/DATASETS.md). No MeterFL, MeasureBench or Cambridge images.

## Own synthetic renders (MIT, this repository)

Rendered by `scripts/generate_synth.py` (`configs/synth_2d.yaml`), val split of
`data/synthetic/synth_2d`; re-saved as JPEG quality 85 at the original 640 x 640 px. The file
name prefix is the example registry tag (`configs/gauges.csv`) whose range and unit the render
uses; the tags are made up for the demo.

| file | render id | range | face tilt | conditions | true value |
| --- | --- | --- | --- | --- | --- |
| PI-101__synth_clean.jpg | 019990 | 0–1.6 MPa | 15° | none | 1.540 MPa |
| PI-301__synth_tilt.jpg | 018988 | 0–10 bar | 50° | none | 6.162 bar |
| PI-301__synth_lowlight.jpg | 018307 | 0–10 bar | 29° | low_light | 3.578 bar |
| PI-301__synth_blur.jpg | 018472 | 0–10 bar | 34° | motion_blur | 6.359 bar |

### Background photos in two renders (CC BY 2.0)

`PI-101__synth_clean.jpg` and `PI-301__synth_blur.jpg` have procedural backgrounds. Two renders
use a background photo from Openverse (checked on 2026-11-24 by replaying their seeds,
`assets/backgrounds/ATTRIBUTION.csv`):

| file | background photo | creator | license |
| --- | --- | --- | --- |
| PI-301__synth_tilt.jpg | ["Newberry Geothermal Lease Project"](https://www.flickr.com/photos/50169152@N06/14493274705) | BLM Oregon & Washington | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| PI-301__synth_lowlight.jpg | ["Broseley Pipeworks - Clay Tobacco Pipe Museum - sign"](https://www.flickr.com/photos/39415781@N06/3809582538) | ell brown | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

Changes: the photo is cropped and resized to 640 x 640 px, a synthetic gauge is drawn over it,
and the render adds tilt (tilt image) or low light and noise (low-light image).
These two photos and their changes are not covered by this repository's MIT license.

## Roboflow photos (CC BY 4.0)

Dataset: **Analog Gauge Reading** by Roboflow workspace **hustlegrad**,
<https://universe.roboflow.com/hustlegrad/analog-gauge-reading-v4tii>, license
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Downloaded on 2026-11-10 as the
export `hustlegrad/gauge-detection-2-2ynzm` version 7
(<https://universe.roboflow.com/hustlegrad/gauge-detection-2-2ynzm/dataset/7>).
Changes: the Roboflow export is already resized to 640 x 640 px (stretched); we re-saved it as
JPEG quality 85 and renamed it. No other edits.

| file | split | original file name (Roboflow export) |
| --- | --- | --- |
| roboflow_small.jpg | valid | 20221116_100231_jpg.rf.f9f23f4cc424cc7e04b14efd96174dd3.jpg |
| roboflow_dim.jpg | test | 20221116_100021_jpg.rf.d52eb813f8d6f89ece02beb4c511659f.jpg |
| roboflow_tilt.jpg | valid | 20221116_095841_jpg.rf.387ba583e4d1badb9ffb8bb8a7f70fd8.jpg |
| roboflow_closeup.jpg | test | 16-09-00-14LAB12CP504-1_jpg.rf.b33fe6bc083072ba37b01e2de69efcb2.jpg |
