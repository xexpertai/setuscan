"""Kokoro af_heart narration (CPU). Run: ~/workspace/.video-venv/bin/python video/tts.py  (inside systemd-run MemoryMax=3G)"""
import json, pathlib, subprocess, numpy as np, soundfile as sf
from kokoro import KPipeline
H = pathlib.Path(__file__).parent; B = H / "build"; (B / "audio").mkdir(parents=True, exist_ok=True)
story = json.load(open(H / "story.json")); pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
SIL = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.02,areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.06,areverse"
import sys, re
SPEED = 0.80; SENT_PAUSE = 0.38
RESPELL = [("OpenCV", "Open C V"), ("p95", "p ninety five"), ("Sift", "S I F T"), ("Lambda", "Lambda")]
def tts_text(t):
    for a, b in RESPELL: t = t.replace(a, b)
    return t
for ln in [l for l in story["lines"] if len(sys.argv) < 2 or l["id"] in sys.argv[1:]]:
    parts = []
    for sent in re.split(r"(?<=[.!?])\s+", tts_text(ln["text"])):
        w = np.concatenate([np.asarray(x) for _, _, x in pipe(sent, voice=story["_meta"]["voice"], speed=SPEED)]); nz = np.where(np.abs(w) > 0.004)[0]; w = w[nz[0]:nz[-1] + 1]  # trim Kokoro's own edge silence so pauses are exact
        parts.append(w); parts.append(np.zeros(int(SENT_PAUSE * 24000), dtype=np.float32))
    a = np.concatenate(parts[:-1])
    quiet = np.abs(a) < 0.004; i = 0; keep = np.ones(len(a), bool)  # cap any internal near-silence at 0.5 s (gate: no silence > 0.6 s)
    while i < len(a):
        if quiet[i]:
            j = i
            while j < len(a) and quiet[j]: j += 1
            if j - i > 0.5 * 24000: keep[i + int(0.5 * 24000):j] = False
            i = j
        else: i += 1
    a = a[keep]
    raw = B / "audio" / f"{ln['id']}_raw.wav"; sf.write(raw, a, 24000, subtype="PCM_16")
    out = B / "audio" / f"{ln['id']}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-af", SIL, str(out)], check=True)
    print(ln["id"], round(sf.info(out).duration, 2), flush=True)
