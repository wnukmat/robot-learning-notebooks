"""Turn the percent-format sources in src/ into executed notebooks in notebooks/.

Usage: tools/build.py [name-substring ...]      (no arguments builds everything)
Each source starts with a markdown cell containing {{BADGES}}, which becomes the Colab and guide links.
"""
import json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; REPO = "wnukmat/robot-learning-notebooks"; SITE = "https://www.matthewwnuk.com/"
GROUPS = {"1": "1_foundations", "2": "2_models", "3": "3_robot_learning", "4": "4_data_and_hardware", "5": "5_practice"}
SETUP = '''# %%
# Setup. On Colab this installs the small helper package; where it is already installed it does nothing.
import importlib.util, subprocess, sys
if importlib.util.find_spec("guide_utils") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/''' + REPO + '''"])
'''

def build(src):
    group = GROUPS[src.stem[0]]; out = ROOT / "notebooks" / group / (src.stem[2:] + ".ipynb"); out.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(); guide = re.search(r"^# GUIDE: (\S+)", text, flags=re.M).group(1); text = re.sub(r"^# GUIDE: .*\n", "", text, flags=re.M)
    gpage, _, ganc = guide.partition("#")                            # the first GUIDE line names the page and the section the notebook belongs to
    rel = out.relative_to(ROOT).as_posix()
    badges = f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{rel}) &nbsp; Guide note: [{gpage}]({SITE}{gpage}.html{'#' + ganc if ganc else ''})"
    text = text.replace("{{BADGES}}", badges).replace("# %% SETUP\n", SETUP)
    tmp = ROOT / ".build" / (src.stem + ".py"); tmp.parent.mkdir(exist_ok=True); tmp.write_text(text)
    t0 = time.time(); subprocess.run([str(ROOT / ".venv/bin/jupytext"), "--to", "ipynb", "--output", str(out), str(tmp)], check=True, capture_output=True)
    r = subprocess.run([str(ROOT / ".venv/bin/jupyter"), "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.kernel_name=rln", "--ExecutePreprocessor.timeout=1500", str(out)], capture_output=True, text=True)
    if r.returncode:
        print("FAILED", src.name, "\n", r.stderr[-1800:]); return False
    nb = json.loads(out.read_text())
    for c in nb["cells"]:                                            # drop library warnings so the saved outputs are clean
        if c["cell_type"] == "code": c["outputs"] = [o for o in c["outputs"] if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}; nb["metadata"].pop("jupytext", None)
    out.write_text(json.dumps(nb, indent=1)); print(f"ok {rel}  {time.time() - t0:.0f}s  {out.stat().st_size // 1024} KB"); return True

if __name__ == "__main__":
    want = sys.argv[1:]; ok = [build(s) for s in sorted((ROOT / "src").glob("*.py")) if not want or any(w in s.name for w in want)]
    sys.exit(0 if all(ok) else 1)
