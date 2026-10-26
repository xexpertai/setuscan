/* Web Worker: loads OpenCV 5 (WebAssembly) and runs the SetuScan agent off the UI thread. */
importScripts("vendor/opencv.js", "setuscan.js");
let cvReady = (async () => {
  let c = self.cv;
  if (c instanceof Promise) c = await c;
  else if (c && typeof c.then === "function") c = await new Promise(r => c.then(x => r(x)));
  else if (!c.Mat) await new Promise(r => { c.onRuntimeInitialized = r; });
  return c;
})();
self.onmessage = async e => {
  const { type, imageData, id } = e.data;
  try {
    const cv = await cvReady;
    if (type === "init") { self.postMessage({ type: "ready", opencv: cv.getBuildInformation().match(/OpenCV (\d+\.\d+\.\d+)/)[1] }); return; }
    const res = SetuScan.runAgent(cv, imageData, { onStep: s => self.postMessage({ type: "step", id, step: s }) });
    const a = res.analysis;
    self.postMessage({ type: "done", id, result: { ...res, analysis: a && { mm_per_px: a.mm_per_px, blur_sigma_px: a.blur_sigma_px, rejected: a.rejected, timings_ms: a.timings_ms,
      view: { unit: a.view.unit, scale: a.view.scale, size: a.view.size }, overlay: a.overlay } } });
  } catch (err) { self.postMessage({ type: "error", id, message: String(err && err.message || err) }); }
};
