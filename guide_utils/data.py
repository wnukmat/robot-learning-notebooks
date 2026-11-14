"""Download the small datasets and checkpoints the notebooks use from the Hugging Face Hub.

Everything is cached by huggingface_hub, so each file is downloaded once.
Sources and licences are listed in the dataset card: https://huggingface.co/datasets/mwnuk/robot-learning-guide-data
"""
import json
import numpy as np

DATA_REPO = "mwnuk/robot-learning-guide-data"
MODEL_REPO = "mwnuk/robot-learning-guide-models"


def fetch(path):
    """Local path of one file from the data repo."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATA_REPO, path, repo_type="dataset")


def fetch_model(path):
    """Local path of one file from the models repo."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, path)


def libero():
    """All 379 LIBERO-10 demonstrations: action (N,7), state (N,8), episode (N,), task (N,), frame (N,). 10 steps per second."""
    return dict(np.load(fetch("libero10/actions_states.npz")))


def libero_tasks():
    """{task index: instruction text}."""
    return {int(k): v for k, v in json.load(open(fetch("libero10/tasks.json"))).items()}


def libero_frames():
    """1,516 frames (4 per episode): thumbs (N,96,96,3), clip (N,512) image embeddings, pixels (N,256), episode, task, frac,
    and text_clip (6,512) embeddings of the queries in text_queries."""
    d = dict(np.load(fetch("libero10/frames.npz"))); d["text_queries"] = json.load(open(fetch("libero10/text_queries.json")))
    return d


def libero_sequences():
    """Six-second clips from 12 episodes, starting at step 20: scene (12,60,128,128,3), wrist (same), episode (12,). 10 frames per second."""
    return dict(np.load(fetch("libero10/sequences.npz")))


def tactile_press():
    """One press seen by a gel camera, a 12x12 pressure array and a wrist force-torque sensor (TaF dataset, MIT)."""
    return dict(np.load(fetch("tactile/press.npz")))


def audio(name):
    """A 16 kHz mono clip as float32: name is 'pour' or 'knock' (public-domain recordings)."""
    return np.load(fetch(f"audio/{name}.npy")), 16000


def digits():
    """The scikit-learn 8x8 digits: x (1797,64) scaled to 0..1, y (1797,). Bundled with scikit-learn, so no download."""
    from sklearn.datasets import load_digits
    d = load_digits(); return (d.data / 16.0).astype("float32"), d.target.astype("int64")
