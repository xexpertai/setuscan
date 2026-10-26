# Real photos: gauge vs SetuScan

REAL photos, gauge readings taken by the entrant (see `docs/REAL-PHOTO-PROTOCOL.md`). The gauge width is compared with SetuScan's **median** width; the **p95** width (used by the severity policy) is shown next to it.

| Photo | Gauge width (mm) | SetuScan median (mm) | SetuScan p95 (mm) | Error median-gauge (mm) | Error (%) | Gauge length (mm) | SetuScan length (mm) | Outcome |
|---|---|---|---|---|---|---|---|---|
| crackA.jpg | 0.70 | 0.78 | 1.42 | 0.080 | 11.4 | 85 | 31 | awaiting_approval |
| crackB.jpg | 1.20 | 0.62 | 1.63 | -0.577 | -48.1 | 65 | 84 | awaiting_approval |
| crackA2.jpg | 0.70 | - | - | - | - | 85 | - | closed |
| crackB2.jpg | 1.20 | 1.13 | 2.22 | -0.074 | -6.2 | 65 | 25 | awaiting_approval |

Marker id 0 size used: 18.0 mm.
Photos: 4; measured: 3; refused or failed (listed above, not in the statistics): 1.
Width error on measured photos (median vs gauge): MAE 0.244 mm, median absolute error 0.08 mm, mean signed bias -0.19 mm, mean absolute percentage error 21.9 %.
Small n: these figures describe these photos only. Gauge readings have their own uncertainty (about half the smallest gauge step).
