"""Assemble the final video: slides + live capture + narration. Run with ~/workspace/.video-venv/bin/python video/build.py"""
import json, pathlib, subprocess, numpy as np, soundfile as sf
H = pathlib.Path(__file__).parent; B = H / "build"; PAUSE = 0.34; TEMPO = 1.13  # AI narration retime (human clip stays 1.0x)
ORDER = ["s1_team", "s2_problem", "LIVE", "s10_arch", "s11_results", "s11b_real", "s12_limits", "s13_end"]
LIVE = ["s3_home", "s4_wide", "s5_approve", "s6_trace", "s7_thin", "s8_blurry", "s9_growth"]
def sh(*a): subprocess.run([str(x) for x in a], check=True, capture_output=True)
def aud(i):
    a, sr = sf.read(B / "audio" / f"{i}.wav"); assert sr == 24000
    return np.concatenate([a, np.zeros(int(PAUSE * sr))])

def despike_live(src, ss, d, out):
    """Re-encode the LIVE take at 30 fps, replacing any frame whose fixed LIVE banner is displaced/corrupted (Playwright screencast glitch) with the previous good frame."""
    W, Hh = 1920, 1080; n = int(d * 30 + 0.5)
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", ss, "-i", src, "-t", str(d), "-vf", "fps=30,scale=1920:1080,format=rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "30", "-i", "-", "-vf", "format=yuv420p", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    ref = None; good = None; fixed = []; i = 0
    while True:
        b = rd.stdout.read(W * Hh * 3)
        if len(b) < W * Hh * 3: break
        f = np.frombuffer(b, np.uint8).reshape(Hh, W, 3); ban = f[:42].astype(np.int16)
        if ref is None: ref = ban.copy()
        if good is not None and np.abs(ban - ref).mean() > 4: fixed.append(round(i / 30, 2)); b = good
        else: good = b
        wr.stdin.write(b); i += 1
    wr.stdin.close(); wr.wait(); rd.wait(); print("despiked banner-glitch frames at", fixed, "of", i, flush=True)
segs = []; chunks = []
(B / "seg").mkdir(exist_ok=True)
team = H / "assets/team_clip.mp4"
if False: pass
for k, name in enumerate(ORDER):
    out = B / f"seg/{k+1:02d}_{name}.mp4"
    if name == "LIVE":
        a = np.concatenate([aud(i) for i in LIVE]); d = len(a) / 24000
        despike_live(B / "live.webm", (B / "live_offset.txt").read_text(), d, out)
    else:
        a = aud(name); d = len(a) / 24000
        sh("ffmpeg", "-y", "-loop", 1, "-i", B / f"slides/{name}.png", "-t", d, "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", 18, out)
    segs.append(out); chunks.append(a)
AI_ONLY = True
sf.write(B / "narr_slow.wav", np.concatenate(chunks), 24000, subtype="PCM_16")
sh("ffmpeg", "-y", "-i", B / "narr_slow.wav", "-af", "atempo=%s" % TEMPO, B / "narr.wav")  # built slow, retimed 1.13x
(B / "concat.txt").write_text("".join(f"file '{s}'\n" for s in segs))
sh("ffmpeg", "-y", "-f", "concat", "-safe", 0, "-i", B / "concat.txt", "-c", "copy", B / "video_slow.mp4")
sh("ffmpeg", "-y", "-i", B / "video_slow.mp4", "-vf", "setpts=PTS/%s,fps=30,format=yuv420p" % TEMPO, "-an", "-c:v", "libx264", "-crf", 18, B / "video_ai.mp4")
if team.exists():  # entrant's own selfie clip: opening segment, own voice, blurred side fill + caption strip
    cap = "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Harsha Shinde, solo builder\\: SetuScan':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=1080-62:box=1:boxcolor=0x0f172acc:boxborderw=14"
    fc = "[0:v]split[a][b];[a]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=40:6,eq=brightness=-0.3[bg];[b]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:0,fps=30,format=yuv420p," + cap
    sh("ffmpeg", "-y", "-i", team, "-filter_complex", fc, "-an", "-c:v", "libx264", "-crf", 18, B / "seg/00_team.mp4")
    sh("ffmpeg", "-y", "-i", team, "-vn", "-ac", 1, "-ar", 24000, B / "seg/00_team.wav")
    w, _ = sf.read(B / "seg/00_team.wav"); dc = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",B/"seg/00_team.mp4"],capture_output=True,text=True).stdout
    ai, _ = sf.read(B / "narr.wav"); sf.write(B / "narr.wav", np.concatenate([w[:int(float(dc)*24000)] if len(w) > float(dc)*24000 else np.concatenate([w, np.zeros(int(float(dc)*24000)-len(w))]), ai]), 24000, subtype="PCM_16")
    (B / "concat2.txt").write_text(f"file '{B/'seg/00_team.mp4'}'\nfile '{B/'video_ai.mp4'}'\n")
    sh("ffmpeg", "-y", "-f", "concat", "-safe", 0, "-i", B / "concat2.txt", "-c", "copy", B / "video_only.mp4")
else: (B / "video_ai.mp4").replace(B / "video_only.mp4")
# loudness: two-pass loudnorm to -14 LUFS, TP -1.5
import re
m = subprocess.run(["ffmpeg", "-nostats", "-i", B / "narr.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", m, re.S).group(0))
ln = f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true"
sh("ffmpeg", "-y", "-i", B / "narr.wav", "-af", ln + ",aresample=48000", B / "narr_norm.wav")
dst = H / "setuscan-demo.mp4"
sh("ffmpeg", "-y", "-i", B / "video_only.mp4", "-i", B / "narr_norm.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", "-metadata", "title=SetuScan demo", dst)
print(dst, subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", dst], capture_output=True, text=True).stdout)
if float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",dst],capture_output=True,text=True).stdout) > 300: print("WARNING: video longer than 300 s gate limit; trim narration")
