# Real-photo validation protocol (for the entrant)

Goal: 5 to 10 real cracks, each with a gauge reading, so SetuScan's millimetres can be compared with a physical measurement. Until this is done, the project states that millimetre accuracy is validated on synthetic scenes only.

## What you need
* `docs/marker-sheet.pdf`, printed at **100 % scale** ("Actual size", not "Fit to page"). Check with a ruler: the printed marker's side must match the size written on the sheet.
* A ruler or a crack-width card (a feeler gauge also works). Note the smallest division, for example 0.1 mm on a crack card.
* A phone camera. Wiping the lens helps more than any setting.

## Shooting each crack
1. Tape or hold the printed marker **flat on the wall, right next to the crack** (within about 10 cm), on the same surface and the same plane. Do not curl the paper.
2. Place the ruler or crack card across the crack (or beside it) so its reading is visible, but **take two photos**: one with the card in view, and one without (SetuScan should measure the crack without the card on it). Name them `crackNN_card.jpg` and `crackNN.jpg`.
3. Hold the phone **parallel to the wall**, about 30 to 60 cm away, so the marker and the crack both fill a good part of the frame. Avoid steep angles.
4. Use **good, even light**. No flash glare on the marker or the crack, no deep shadow across them. Tap to focus on the crack and wait for the focus to settle; keep still (brace on the wall) to avoid blur.
5. Take 5 to 10 different cracks. Include at least one thin crack (under 0.5 mm) and one wide crack (over 1 mm) if you can find them. Include a bad photo or two (a bit blurry, no marker) on purpose: failures are reported, not hidden.
6. Read the gauge at the **widest point you can read reliably** and write it down in mm, plus the visible crack length in mm if you measured it.

## Gauge CSV
Create `gauges.csv` next to the photos:

```
file,gauge_width_mm,gauge_length_mm
crack01.jpg,0.6,180
crack02.jpg,1.5,
```

`gauge_length_mm` may be empty. Use the photo **without** the card in `file`.

## Run
```
cd ~/workspace/hk-opencv
PYTHONPATH=. .venv/bin/python eval/eval_real_photos.py /path/to/photos /path/to/gauges.csv
```
This writes `eval/results/real_photos.json` and `eval/results/real_photos.md` (gauge vs SetuScan table). The gauge width is compared with SetuScan's median width, with p95 shown next to it. Photos that SetuScan refuses to measure are listed with the reason and left out of the error statistics.

## Rules for using the result
* Report whatever comes out, including large errors and refusals. Do not drop photos after seeing the numbers.
* Label the data REAL, with the number of photos, the device and the gauge type.
* Remove people, faces, number plates and location details from photos before publishing any of them.
* Only then add the table to the README, the technical report (section 6) and the Devpost text, and change "no real-photo validation" to the actual finding.
