import json, sys, pathlib, numpy as np
from PIL import Image
F = pathlib.Path(sys.argv[1]); idx = json.load(open(F / "index.json"))
def g(f): return np.asarray(Image.open(F / f).convert("L").resize((96, 54)), dtype=float)
groups = []; 
for e in idx:
    a = g(e["file"])
    if groups:
        d = float(np.abs(a - groups[-1]["a"]).mean())
        if d < 2.0: groups[-1]["members"].append((e["t"], round(d, 2))); continue
    groups.append(dict(t=e["t"], file=e["file"], a=a, members=[]))
print(len(idx), "frames ->", len(groups), "groups")
json.dump([dict(t=x["t"], file=x["file"], members=x["members"]) for x in groups], open(sys.argv[2], "w"))
