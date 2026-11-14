"""Write README.md from the notebook sources."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; REPO = "wnukmat/robot-learning-notebooks"; SITE = "https://www.matthewwnuk.com/"
GROUPS = {"1": ("1_foundations", "Foundations"), "2": ("2_models", "Models"), "3": ("3_robot_learning", "Robot learning"), "4": ("4_data_and_hardware", "Data and hardware"), "5": ("5_practice", "Practice")}
rows = {}
for src in sorted((ROOT / "src").glob("*.py")):
    t = src.read_text(); title = re.search(r"^# # (.+)$", t, flags=re.M).group(1); guide = re.search(r"^# GUIDE: (\S+)", t, flags=re.M).group(1)
    body = t.split("{{BADGES}}", 1)[1]; para = re.search(r"#\n((?:# .+\n)+)", body).group(1); summary = " ".join(l[2:] for l in para.splitlines()); summary = re.split(r"(?<=[.])\s", summary)[0]
    folder, _ = GROUPS[src.stem[0]]; rel = f"notebooks/{folder}/{src.stem[2:]}.ipynb"
    rows.setdefault(src.stem[0], []).append(f"| [{title}]({rel}) | {summary} | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{rel}) | [note]({SITE}{guide}.html) |")
n = sum(len(v) for v in rows.values())
out = f"""# Robot learning notebooks

{n} runnable notebooks that accompany [Deep learning and robotics: a working guide]({SITE}guide.html).
Each one reproduces the measurements behind a note in the guide, on a problem small enough to run in a
few minutes, and ends with things to change and try.

Every notebook runs on a laptop or a free Colab session. No GPU is needed, though a few are faster with one.

## Notebooks
"""
for k, (folder, name) in GROUPS.items():
    out += f"\n### {name}\n\n| Notebook | What it shows | Run | Guide |\n|---|---|---|---|\n" + "\n".join(rows.get(k, [])) + "\n"
out += f"""
## Running them

**On Colab.** Click a badge above. The first cell installs the small helper package from this repository.

**Locally.**

```bash
git clone https://github.com/{REPO}
cd robot-learning-notebooks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
jupyter lab
```

## Data and models

The notebooks download a few small files on first use and cache them.

- **Data:** [mwnuk/robot-learning-guide-data](https://huggingface.co/datasets/mwnuk/robot-learning-guide-data).
  Actions, states and sample frames from the LIBERO-10 demonstrations, one tactile press, and two short sound
  clips. Each is derived from an openly licensed source, listed with its licence in the dataset card.
- **Models:** [mwnuk/robot-learning-guide-models](https://huggingface.co/mwnuk/robot-learning-guide-models).
  Small checkpoints for the two notebooks that would otherwise spend minutes training. Both have a
  `TRAIN = True` switch that reproduces them.

The control experiments use a toy task defined in [`guide_utils/toy.py`](guide_utils/toy.py): a point robot
that must reach a target without touching an obstacle. It is deliberately simple, so that every claim can be
checked by running a policy in the loop in seconds. Results on it illustrate ideas. They are not results
about real robots.

## Layout

```
notebooks/     the notebooks, saved with their outputs
src/           the same notebooks as plain Python files (jupytext percent format)
guide_utils/   plot style, data loading, the toy control task and a grid world
tools/         build.py turns src/ into executed notebooks
```

To change a notebook, edit its file in `src/` and run `python tools/build.py <name>`.

## Licence

MIT. The data keeps the licences of its sources.
"""
(ROOT / "README.md").write_text(out); print(n, "notebooks listed")
