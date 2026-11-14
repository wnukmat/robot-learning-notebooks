# Robot learning notebooks

27 runnable notebooks that accompany [Deep learning and robotics: a working guide](https://www.matthewwnuk.com/guide.html).
Each one reproduces the measurements behind a note in the guide, on a problem small enough to run in a
few minutes, and ends with things to change and try.

Every notebook runs on a laptop or a free Colab session. No GPU is needed, though a few are faster with one.

## Notebooks

### Foundations

| Notebook | What it shows | Run | Guide |
|---|---|---|---|
| [The training loop, from scratch](notebooks/1_foundations/01_training_loop.ipynb) | Every model in the guide is trained by the same four steps: forward pass, loss, backward pass, update. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/01_training_loop.ipynb) | [note](https://www.matthewwnuk.com/guide-foundations.html) |
| [Learning rate and optimisers](notebooks/1_foundations/02_learning_rate_and_optimisers.ipynb) | The learning rate is the most important number in training. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/02_learning_rate_and_optimisers.ipynb) | [note](https://www.matthewwnuk.com/guide-foundations.html) |
| [Why deep networks train at all](notebooks/1_foundations/03_depth_and_residuals.ipynb) | Stacking plain layers stops working quickly. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/03_depth_and_residuals.ipynb) | [note](https://www.matthewwnuk.com/guide-foundations.html) |
| [How much of a result is noise](notebooks/1_foundations/04_seeds_and_noise.ipynb) | Before asking whether a change helped, ask how much the number moves when nothing changes. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/04_seeds_and_noise.ipynb) | [note](https://www.matthewwnuk.com/guide-things-i-know.html) |
| [Sequence to sequence: the bottleneck and how attention removes it](notebooks/1_foundations/05_sequence_to_sequence.ipynb) | Three models learn the same task: read a sequence of symbols and write it reversed. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/05_sequence_to_sequence.ipynb) | [note](https://www.matthewwnuk.com/guide-sequence-to-sequence.html) |
| [Attention from scratch](notebooks/1_foundations/06_attention_from_scratch.ipynb) | Attention is one operation, and every variant in the guide is a change to where its inputs come from or to its mask. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/1_foundations/06_attention_from_scratch.ipynb) | [note](https://www.matthewwnuk.com/guide-attention.html) |

### Models

| Notebook | What it shows | Run | Guide |
|---|---|---|---|
| [Linear probes: what do frozen features know?](notebooks/2_models/01_linear_probes.ipynb) | A representation is useful if simple things can be read from it with little data. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/2_models/01_linear_probes.ipynb) | [note](https://www.matthewwnuk.com/guide-representation-learning.html) |
| [Contrastive losses: triplet, hard mining, and InfoNCE](notebooks/2_models/02_contrastive_losses.ipynb) | A contrastive loss is told only which items belong together. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/2_models/02_contrastive_losses.ipynb) | [note](https://www.matthewwnuk.com/guide-contrastive-learning.html) |
| [Retrieval on robot data](notebooks/2_models/03_retrieval.ipynb) | Retrieval answers a question by finding the nearest items in a collection. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/2_models/03_retrieval.ipynb) | [note](https://www.matthewwnuk.com/guide-retrieval.html) |
| [Flow matching against diffusion, on a problem small enough to see](notebooks/2_models/04_flow_matching_vs_diffusion.ipynb) | Two copies of one small network learn the same two-dimensional dataset. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/2_models/04_flow_matching_vs_diffusion.ipynb) | [note](https://www.matthewwnuk.com/guide-generative-modeling.html) |
| [Low-rank adaptation](notebooks/2_models/05_low_rank_adaptation.ipynb) | Full fine-tuning updates every weight. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/2_models/05_low_rank_adaptation.ipynb) | [note](https://www.matthewwnuk.com/guide-foundation-models.html) |

### Robot learning

| Notebook | What it shows | Run | Guide |
|---|---|---|---|
| [Frames, kinematics and rotations](notebooks/3_robot_learning/01_kinematics.ipynb) | The classical layer that every learned policy sits on, worked through for a two-link arm: transforms that chain, forward and inverse kinematics, the Jacobian and what happens near a singularity, and why a rotation should not be written as a single angle. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/01_kinematics.ipynb) | [note](https://www.matthewwnuk.com/guide-robotics-fundamentals.html) |
| [Controllers: from a target to motor torque](notebooks/3_robot_learning/02_controllers.ipynb) | A learned policy outputs targets tens of times a second. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/02_controllers.ipynb) | [note](https://www.matthewwnuk.com/guide-robotics-fundamentals.html) |
| [Behaviour cloning and compounding error](notebooks/3_robot_learning/03_behaviour_cloning.ipynb) | Behaviour cloning is supervised learning: observations in, the demonstrator's actions out. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/03_behaviour_cloning.ipynb) | [note](https://www.matthewwnuk.com/guide-robot-learning-basics.html) |
| [Action spaces and normalisation, on real robot data](notebooks/3_robot_learning/04_action_spaces.ipynb) | What exactly does a policy output? This notebook looks at 379 real demonstrations to show the same motion written as absolute positions and as relative steps, how a relative action relates to the motion it produced, why actions must be normalised, and why step-to-step deltas drift. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/04_action_spaces.ipynb) | [note](https://www.matthewwnuk.com/guide-robot-learning-basics.html) |
| [Turning actions into tokens](notebooks/3_robot_learning/05_action_tokenisation.ipynb) | A language model outputs tokens, so a policy built on one has to write actions as tokens. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/05_action_tokenisation.ipynb) | [note](https://www.matthewwnuk.com/guide-robot-policies.html) |
| [Reinforcement learning on a grid](notebooks/3_robot_learning/06_reinforcement_learning.ipynb) | Reinforcement learning has no demonstrator. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/06_reinforcement_learning.ipynb) | [note](https://www.matthewwnuk.com/guide-reinforcement-learning.html) |
| [The baseline every world model must beat](notebooks/3_robot_learning/07_world_model_baselines.ipynb) | A world model predicts future camera frames. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/3_robot_learning/07_world_model_baselines.ipynb) | [note](https://www.matthewwnuk.com/guide-world-models.html) |

### Data and hardware

| Notebook | What it shows | Run | Guide |
|---|---|---|---|
| [Inside a robot dataset](notebooks/4_data_and_hardware/01_inside_a_robot_dataset.ipynb) | What is actually in a demonstration dataset? This notebook opens LIBERO-10, the dataset used for figures throughout the guide: 379 demonstrations of ten tasks. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/4_data_and_hardware/01_inside_a_robot_dataset.ipynb) | [note](https://www.matthewwnuk.com/guide-robot-data.html) |
| [Scoring data quality](notebooks/4_data_and_hardware/02_data_quality.ipynb) | Past a few hundred episodes nobody can watch everything, so quality is scored by program and reviewed by exception. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/4_data_and_hardware/02_data_quality.ipynb) | [note](https://www.matthewwnuk.com/guide-data-quality.html) |
| [Data efficiency: learning curves, a power law, and what pretraining buys](notebooks/4_data_and_hardware/03_data_efficiency.ipynb) | How much data does a policy need? The honest answer is a curve. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/4_data_and_hardware/03_data_efficiency.ipynb) | [note](https://www.matthewwnuk.com/guide-data-efficiency.html) |
| [The data flywheel](notebooks/4_data_and_hardware/04_data_flywheel.ipynb) | A deployed policy's failures show where its data is thin. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/4_data_and_hardware/04_data_flywheel.ipynb) | [note](https://www.matthewwnuk.com/guide-data-flywheel.html) |
| [Touch, force and sound](notebooks/4_data_and_hardware/05_touch_force_and_sound.ipynb) | Cameras lead until the hand closes. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/4_data_and_hardware/05_touch_force_and_sound.ipynb) | [note](https://www.matthewwnuk.com/guide-senses.html) |

### Practice

| Notebook | What it shows | Run | Guide |
|---|---|---|---|
| [Evaluation statistics for small numbers of trials](notebooks/5_practice/01_evaluation_statistics.ipynb) | Robot evaluations are slow, so the number of trials is small, and small samples carry wide error bars. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/5_practice/01_evaluation_statistics.ipynb) | [note](https://www.matthewwnuk.com/guide-evaluation.html) |
| [Failure triage on real data](notebooks/5_practice/02_failure_triage.ipynb) | When a robot fails, find the layer at fault before retraining anything. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/5_practice/02_failure_triage.ipynb) | [note](https://www.matthewwnuk.com/guide-failure-triage.html) |
| [Quantising a policy until it breaks](notebooks/5_practice/03_quantisation.ipynb) | Quantisation stores weights in fewer bits. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/5_practice/03_quantisation.ipynb) | [note](https://www.matthewwnuk.com/guide-deployment.html) |
| [Calibration, and knowing when not to trust a model](notebooks/5_practice/04_calibration_and_probing.ipynb) | A model that says "95% sure" should be right 95% of the time. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wnukmat/robot-learning-notebooks/blob/main/notebooks/5_practice/04_calibration_and_probing.ipynb) | [note](https://www.matthewwnuk.com/guide-understanding-and-safety.html) |

## Running them

**On Colab.** Click a badge above. The first cell installs the small helper package from this repository.

**Locally.**

```bash
git clone https://github.com/wnukmat/robot-learning-notebooks
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
