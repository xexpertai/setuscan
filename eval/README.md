# Evaluation

Every number quoted in the docs comes from a JSON file in `eval/results/`, written by a script in
this folder. Each file records its provenance: git commit, OpenCV/numpy/Python versions, time.
Re-run everything with:

```bash
uv run python scripts/download_data.py       # KrakN from Zenodo (CC BY 4.0), MD5-checked
uv run python eval/tune_krakn_dev.py         # optional: parameter sweep on the DEV split only (~9 min)
uv run python eval/eval_krakn.py             # crack / no-crack on the held-out TEST split (~1 min)
uv run python eval/eval_synthetic_mm.py      # mm accuracy on SYNTHETIC scenes (~3 min)
uv run python eval/eval_agent.py             # agent scenarios on SYNTHETIC scenes (~2 min)
```

| File | What it measures | Data |
|---|---|---|
| `krakn_dev_tuning.json` | grid search of 4 detector parameters, scored on dev only | KrakN dev split (30 %) |
| `krakn_classification.json`, `krakn_scores.csv` | precision, recall, F1, accuracy, AP, threshold sweep, runtime | KrakN test split (70 %), never used for tuning |
| `krakn_false_positives.jpg`, `krakn_false_negatives.jpg` | worst test failures with the detector's mask | KrakN test split |
| `synthetic_mm_accuracy.json` | marker detection, width error (before and after blur correction), length error, by true width and camera resolution | synthetic scenes with exact ground truth |
| `agent_scenarios.json` | task success, terminal-status accuracy, false or missed escalations, steps, time, every failure | 14 synthetic scenarios × 5 seeds |

## Datasets and licences

* **KrakN associated dataset**: Zarski, Wojcik, Miszczak (2021). Zenodo record 5266356,
  DOI [10.5281/zenodo.5266356](https://doi.org/10.5281/zenodo.5266356), licence
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (read from the Zenodo record
  metadata on 11 Oct 2026). These are 224×224 px patches of concrete bridge surfaces labelled
  Crack (8,057 files) or Background (8,057 files). Files ending `_r` (4,007) are rotated copies of
  crack patches and are excluded, leaving 4,050 crack and 8,057 background images. The failure
  montages in `results/` are derived from this dataset (CC BY 4.0, same attribution).
* **SDNET2018** (Utah State University, DOI 10.15142/T3TD19). Its dataset page states CC BY 4.0,
  but the file host answers scripted downloads with a browser challenge, so it was **not**
  downloaded or used. It can be added later by downloading it by hand.
* **Synthetic scenes** (`setuscan/synth.py`) are rendered here with exact ground truth. They are
  labelled SYNTHETIC everywhere and are not evidence of real-world accuracy.

## Protocol notes and honest caveats

* KrakN labels are image-level, so this evaluates detection, not pixel segmentation or mm width.
  The patches carry no scale marker, so no mm accuracy can be measured on them.
* The KrakN split is a fixed hash of the file name. Four detector parameters and one score
  threshold were chosen on dev, and test was scored once with those settings.
* mm accuracy is shown only on synthetic images. A real-photo check (marker photos with
  ruler or crack-gauge readings) is still to be done and is listed as a limitation.
* The agent scenarios use the deterministic rules decider. They test that OpenCV outputs drive
  the intended next action. They do not test an LLM.
