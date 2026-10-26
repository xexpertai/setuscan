"""Record the LIVE site (one continuous 1920x1080 take) with scripted actions timed to the narration segments."""
import pathlib, shutil, sys, time, wave
from playwright.sync_api import sync_playwright
H = pathlib.Path(__file__).parent; B = H / "build"; PAUSE = 0.34
URL = "https://l4734hezca.execute-api.ap-southeast-2.amazonaws.com"
IDS = ["s3_home", "s4_wide", "s5_approve", "s6_trace", "s7_thin", "s8_blurry", "s9_growth"]
def dur(i):
    with wave.open(str(B / "audio" / f"{i}.wav")) as w: return w.getnframes() / w.getframerate()
D = {i: dur(i) + PAUSE for i in IDS}
BANNER = "document.body.style.paddingTop='46px';const b=document.createElement('div');b.textContent='LIVE: %s  |  AWS Lambda, ap-southeast-2  |  OpenCV 5.0.0';b.style.cssText='position:fixed;top:0;left:0;right:0;height:42px;background:#0f172a;color:#e2e8f0;font:600 22px DejaVu Sans,Arial,sans-serif;padding:7px 20px;z-index:99999';document.body.appendChild(b)" % URL
t0 = 0.0
def until(T):
    r = T - (time.monotonic() - t0)
    if r > 0: time.sleep(r)
def glide(pg, sel, secs, off=70):
    pg.evaluate("""([s,secs,off])=>new Promise(res=>{const e=document.querySelector(s);const y1=e.getBoundingClientRect().top+scrollY-off;const y0=scrollY;const t=performance.now();
      const f=n=>{const k=Math.min(1,(n-t)/(secs*1000));const q=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;scrollTo(0,y0+(y1-y0)*q);k<1?requestAnimationFrame(f):res()};requestAnimationFrame(f)})""", [sel, secs, off])
def top(pg): pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
def stable(pg, tries=40):
    """Wait for a stable repaint: two consecutive identical full-viewport screenshots (no half-painted panel in the recording)."""
    prev = None
    for _ in range(tries):
        pg.evaluate("()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
        cur = pg.screenshot(type="jpeg", quality=40)
        if cur == prev: return
        prev = cur
        pg.wait_for_timeout(120)
def run_sample(pg, sid):
    pg.click(f'button[data-sample="{sid}"]')
    try: pg.wait_for_selector("#busy", state="visible", timeout=4000)
    except Exception: pass
    pg.wait_for_selector("#busy", state="hidden", timeout=60000); pg.wait_for_selector("#result:not([hidden])", timeout=10000); pg.wait_for_function("()=>{const i=document.querySelector('#view');return i.complete&&i.naturalWidth>0}", timeout=30000); stable(pg)
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(B / "rec"), record_video_size={"width": 1920, "height": 1080}, bypass_csp=True)
    tv = time.monotonic(); pg = ctx.new_page(); t0 = time.monotonic()
    starts = {}; T = 0.0
    for i in IDS: starts[i] = T; T += D[i]
    ends = {i: starts[i] + D[i] for i in IDS}
    pg.set_default_timeout(90000); pg.goto(URL + "/", wait_until="networkidle", timeout=90000); pg.evaluate(BANNER); pg.wait_for_selector('button[data-sample="wide"]'); pg.wait_for_timeout(1500); t0 = time.monotonic(); (B / "live_offset.txt").write_text(str(round(t0 - tv, 2)))
    # s3 home
    glide(pg, "#h-samples", 4, 60); until(starts["s3_home"] + D["s3_home"] * 0.55); glide(pg, "#h-upload", 3, 120); until(ends["s3_home"] - 0.2)
    # s4 wide
    glide(pg, "#h-samples", 1.2, 60); run_sample(pg, "wide"); pg.wait_for_timeout(600); glide(pg, "#result", 2, 60)
    until(starts["s4_wide"] + D["s4_wide"] * 0.6); glide(pg, "#meas", 2.5, 120); until(ends["s4_wide"] - 0.2)
    # s5 approve
    glide(pg, "#approval-card", 1.5, 100); pg.click("#rev"); pg.keyboard.type("Harsha Shinde (live demo)", delay=45); pg.wait_for_timeout(500)
    pg.click('button[data-act="approve"]'); pg.wait_for_selector("#ticket-card:not([hidden])", timeout=30000); pg.wait_for_timeout(500); glide(pg, "#ticket-card", 1.5, 140); until(ends["s5_approve"] - 0.2)
    # s6 trace
    glide(pg, "#trace", 2.5, 90); until(starts["s6_trace"] + D["s6_trace"] * 0.55)
    pg.evaluate("(()=>{const t=document.querySelector('#trace');t.scrollIntoView({block:'start'});})()"); pg.evaluate("window.scrollBy({top:420,behavior:'smooth'})"); until(ends["s6_trace"] - 0.2)
    # s7 thin
    top(pg); pg.wait_for_timeout(900); run_sample(pg, "thin"); pg.wait_for_timeout(400); glide(pg, "#result", 1.5, 60); until(starts["s7_thin"] + D["s7_thin"] * 0.5); glide(pg, "#trace", 3, 90); until(ends["s7_thin"] - 0.2)
    # s8 blurry
    top(pg); pg.wait_for_timeout(900); run_sample(pg, "blurry"); pg.wait_for_timeout(400); glide(pg, "#result", 1.5, 60); until(starts["s8_blurry"] + D["s8_blurry"] * 0.55); glide(pg, "#trace", 2.5, 90); until(ends["s8_blurry"] - 0.2)
    # s9 growth
    top(pg); pg.wait_for_timeout(900); run_sample(pg, "growth"); pg.wait_for_timeout(400)
    pg.click('button[data-img="comparison"]'); pg.wait_for_function("()=>{const i=document.querySelector('#view');return i.complete&&i.naturalWidth>0&&i.dataset.want==='comparison'}", timeout=30000); stable(pg)
    glide(pg, "#result", 1.5, 60); until(starts["s9_growth"] + D["s9_growth"] * 0.62); glide(pg, "#comp", 2.5, 140); until(ends["s9_growth"] - 0.1)
    print("elapsed", round(time.monotonic() - t0, 1), "target", round(T, 1))
    vp = pg.video.path(); ctx.close(); b.close()
    shutil.copy(vp, B / "live.webm"); print(vp)
