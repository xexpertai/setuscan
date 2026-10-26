"""Render the non-live slides to build/slides/*.png at 1920x1080 (run with a python that has playwright)."""
import pathlib
from playwright.sync_api import sync_playwright
H = pathlib.Path(__file__).parent; O = H / "build/slides"; O.mkdir(parents=True, exist_ok=True)
CSS = """*{box-sizing:border-box}body{margin:0;width:1920px;height:1080px;background:#0f172a;color:#f1f5f9;font-family:'DejaVu Sans',Arial,sans-serif;padding:80px 110px;position:relative}
h1{font-size:84px;margin:0 0 30px}h2{font-size:64px;margin:0 0 36px}p,li{font-size:40px;line-height:1.4;color:#cbd5e1}.tag{color:#38bdf8}
table{border-collapse:collapse;font-size:34px;width:100%}td,th{padding:16px 20px;border-bottom:2px solid #334155;text-align:left}th{color:#38bdf8}
.foot{position:absolute;left:110px;right:110px;bottom:40px;font-size:26px;color:#94a3b8}.big{font-size:58px;color:#fff}.warn{color:#fbbf24}.ok{color:#4ade80}"""
import base64, json
def b64(path):
    return "data:image/%s;base64,%s" % ("png" if str(path).endswith("png") else "jpeg", base64.b64encode(pathlib.Path(path).read_bytes()).decode())
VIS = H / "build/vis"; ROOT = H.parent
def fig(path, cap, style="", h=None, pos="center"):
    ih = f"height:{h}px;object-fit:cover;object-position:{pos};" if h else ""
    return f"<figure style='margin:0;{style}'><img src='{b64(path)}' style='width:100%;{ih}display:block;border:3px solid #334155;border-radius:10px'><figcaption>{cap}</figcaption></figure>"
CSS += "figcaption{font-size:26px;color:#94a3b8;margin-top:10px}.cols{display:flex;gap:60px}.col{flex:1}.card{background:#1e293b;border-radius:14px;padding:26px 34px}pre{font:26px/1.45 'DejaVu Sans Mono',monospace;color:#e2e8f0;margin:0;white-space:pre-wrap}.k{color:#38bdf8}"
S = {}
# team slide: honest team card; the real team clip (video/assets/team_clip.mp4) is prepended by build.py when it exists
S["s1_team"] = ("<div class=cols><div class=col style='flex:1.05'><h1>SetuScan</h1><p class=tag style='font-size:44px'>Agentic crack inspection with OpenCV 5 on AWS</p>"
  "<div class=card style='margin-top:60px'><p class=big style='margin:0 0 8px'>Team: Harsha Shinde (solo)</p><p style='margin:0'>Role: designer, developer, evaluator and presenter of this entry</p>"
  "<p style='font-size:30px;margin:14px 0 0'>Built with an AI coding assistant (Claude Code), disclosed in the README</p></div>"
  "<p style='margin-top:40px;font-size:34px'>OpenCV AI Competition 2026, powered by AWS</p></div>"
  "<div class=col style='flex:0.8'>" + fig(H / "assets/team_photo.jpg", "Harsha Shinde, solo builder: SetuScan", "", 760, "top") + "</div></div>"
  "<div class=foot>Narration: AI text-to-speech voice (Kokoro).</div>")
S["s2_problem"] = ("<h2>The problem</h2><div class=cols><div class=col style='flex:1.1'><ul style='margin-top:0'><li>Cracks are recorded by eye: a photo, maybe a pencil mark.</li><li>A phone photo has no scale, so width in mm is unknown.</li><li>Growth between visits is hard to judge by eye.</li><li>Findings must reach an engineer, who stays in charge.</li></ul>"
  "<p class=tag style='margin-top:40px'>SetuScan: measured, comparable over time, auditable.</p></div>"
  "<div class=col style='display:flex;flex-direction:column;gap:14px'>" + fig(ROOT / "web/samples/nomarker.jpg", "Without a marker: no scale, pixels only (SYNTHETIC)", "", 200, "right") + fig(VIS / "wide_current_overlay.jpg", "With the printed marker: millimetres, working view (SYNTHETIC)", "width:700px", 500, "top") + "</div></div>"
  "<div class=foot>Decision support only. Not a structural certification. Severity bands are an EXAMPLE policy.</div>")
S["s10_arch"] = "<img src='%s' style='position:absolute;left:0;top:0;width:1920px;height:1080px;object-fit:contain;background:#f8fafc'>" % b64(H.parent / "docs/architecture.png")
acc = json.load(open(ROOT / "eval/results/synthetic_mm_accuracy.json"))["by_width_mm"]
mx = max(v["work_width_mae_mm"] for v in acc.values())
bars = "".join(f"<div style='display:flex;align-items:center;margin:7px 0;font-size:26px'><span style='width:150px'>{w} mm</span><span style='display:inline-block;height:26px;width:{int(520*v['work_width_mae_mm']/mx)}px;background:#38bdf8;border-radius:4px'></span><span style='margin-left:14px'>{v['work_width_mae_mm']:.3f}</span></div>" for w, v in acc.items())
S["s11_results"] = ("<h2 style='font-size:54px;margin-bottom:20px'>Results (all from eval/ files)</h2><table style='font-size:27px'>"
  "<tr><th>Evaluation</th><th>Data</th><th>Result</th></tr>"
  "<tr><td>Crack detection</td><td>KrakN public test split, 2,819 / 5,677 patches</td><td>F1 0.868 (P 0.852, R 0.884)</td></tr>"
  "<tr><td>Width accuracy (median width)</td><td>SYNTHETIC scenes, exact truth, n=84</td><td>MAE 0.054 mm</td></tr>"
  "<tr><td>Agent task success</td><td>SYNTHETIC scenarios, 14 x 5 seeds</td><td>69 / 70</td></tr>"
  "<tr><td>LIVE AWS endpoint</td><td>6 labelled samples, 15 runs</td><td>6 / 6 expected outcomes</td></tr>"
  "<tr><td>LIVE real uploads</td><td>one photo / two visits, India to Sydney</td><td>4.4 s / 8.5 s</td></tr></table>"
  "<div class=cols style='margin-top:26px;align-items:flex-start'><div class=col style='flex:0.9'><p style='font-size:28px;color:#38bdf8;margin:0 0 6px'>Width error by true width (MAE, mm, SYNTHETIC)</p>" + bars + "</div>"
  "<div class=col style='flex:0.55'>" + fig(ROOT / "eval/results/krakn_false_positives.jpg", "KrakN false positives", "", 300, "top") + "</div><div class=col style='flex:0.55'>" + fig(ROOT / "eval/results/krakn_false_negatives.jpg", "KrakN false negatives", "", 300, "top") + "</div></div>"
  "<div class=foot>Synthetic data is labelled as such. Real-photo check is preliminary (next slide). Accuracy = median width; the policy uses the p95 width, which reads higher. Files: eval/results/*.json, eval/live-aws.md</div>")
tr = json.load(open(VIS / "trace_wide.json"))
trace = "\n".join(f"<span class=k>{i+1}. {t['action']}</span>\n   {t['rationale']}" for i, t in enumerate(tr))
rp = json.load(open(ROOT / "eval/results/real_photos.json")); rows = {r["file"]: r for r in rp["rows"]}; sm = rp["summary"]
def rrow(f):
    r = rows[f]; m = r.get("setuscan_width_median_mm")
    if m is None: return f"<tr><td>{f}</td><td>{r['gauge_width_mm']:.1f} mm</td><td>-</td><td>-</td><td class=warn>no crack found (miss)</td></tr>"
    return f"<tr><td>{f}</td><td>{r['gauge_width_mm']:.1f} mm</td><td>{m:.2f} mm</td><td>{r['width_err_pct']:+.1f} %</td><td class=ok>measured</td></tr>"
S["s11b_real"] = ("<h2 style='font-size:54px;margin-bottom:18px'>Real-photo check (preliminary)</h2><div class=cols style='align-items:flex-start'><div class=col style='flex:1.25'>"
  "<table style='font-size:25px'><tr><th>REAL photo</th><th>Gauge width (ruler, max)</th><th>SetuScan median width</th><th>Error</th><th>Outcome</th></tr>" + "".join(rrow(f) for f in ["crackA.jpg","crackB.jpg","crackA2.jpg","crackB2.jpg"]) + "</table>"
  f"<ul style='font-size:26px;margin-top:12px;line-height:1.3'><li>Marker printed at ~{rp['marker_id0_mm_used']:.0f} mm, measured with the ruler; scale set accordingly</li><li class=warn>n = {sm['n_photos']} photos of 2 cracks; preliminary, not an accuracy claim</li><li>Crack length not reliable on rough plaster</li></ul></div>"
  "<div class=col style='flex:1;display:flex;gap:18px'>" + fig(ROOT / "eval/real_photos/crackB2.jpg", "crackB2: measured (REAL photo)", "flex:1", 440, "center") + fig(ROOT / "eval/real_photos/crackA2.jpg", "crackA2: missed (REAL photo)", "flex:1", 440, "center") + "</div></div>"
  "<div class=foot>Source: eval/results/real_photos.json. Gauge readings by the entrant. Small n: describes these photos only.</div>")
S["s12_limits"] = ("<h2>Honest limits</h2><div class=cols><div class=col style='flex:1.1'><ul style='margin-top:0;font-size:34px'><li class=warn>Real-photo check is preliminary: 4 photos of 2 cracks (previous slide). Main accuracy is on synthetic scenes.</li><li>Thin cracks (0.3 mm and below) are over-read.</li><li>Severity bands are an example policy.</li><li class=warn>Decisions come from deterministic rules plus a human approval gate. No LLM is used (Amazon Bedrock access was not available on this account).</li><li>UI is served directly by Lambda (CloudFront needs account verification).</li><li>Built with an AI coding assistant (Claude Code), disclosed in the README.</li></ul></div>"
  "<div class=col style='flex:0.9'><p style='font-size:28px;color:#38bdf8;margin:0 0 10px'>Every decision is traced (wide sample, rules decider)</p><div class=card><pre>" + trace + "</pre></div></div></div>")
S["s13_end"] = ("<div class=cols><div class=col><h1>SetuScan</h1><p class=big>Harsha Shinde</p><p>Code: github.com/xexpertai/setuscan</p><p>Live: l4734hezca.execute-api.ap-southeast-2.amazonaws.com</p>"
  "<p style='font-size:30px;color:#94a3b8;margin-top:50px'>Narration: AI text-to-speech voice (Kokoro). Code and docs written with Claude Code under the entrant's direction. Samples are synthetic or CC BY 4.0 (KrakN). Decision support only. Built with OpenCV 5 and AWS.</p></div>"
  "<div class=col style='flex:0.8'>" + fig(VIS / "growth_comparison.jpg", "Growth between two visits: red = new crack (SYNTHETIC)", "margin-top:30px") + "</div></div>")
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
    for k, v in S.items():
        pg.set_content(f"<style>{CSS}</style>{v}"); pg.wait_for_timeout(300); pg.screenshot(path=str(O / f"{k}.png")); print(k)
    b.close()
