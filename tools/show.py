"""Print the text outputs of a notebook, and save its images to .build/img for a quick look."""
import json, sys, base64
from pathlib import Path
for f in sys.argv[1:]:
    nb = json.loads(Path(f).read_text()); out = Path(".build/img"); out.mkdir(parents=True, exist_ok=True); k = 0; print("==", f)
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream": print("".join(o["text"]).rstrip()[:700])
            if "data" in o and "image/png" in o["data"]:
                (out / f"{Path(f).stem}_{k}.png").write_bytes(base64.b64decode(o["data"]["image/png"])); k += 1
            if o.get("output_type") == "error": print("ERROR", o["ename"], o["evalue"][:300])
    print(f"[{k} figures]")
