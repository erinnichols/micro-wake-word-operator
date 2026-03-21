# Custom micro-wake-word models in ESPHome

A fork of the [microWakeWord basic training notebook](https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb), adapted for local training on a **Windows PC with an NVIDIA GPU via WSL2** and tuned for the wake word **"hey frank"**.

[<img src="/media/device_setup_example.jpg" width="400" alt="image of Homeassistant voice assistant setup" />](/media/device_setup_example.jpg)

### What's different from the original notebook

- **Confusable negative generation** — TTS clips of phonetically similar phrases ("hey fran", "hey finn", etc.) trained as high-penalty hard negatives to reduce false triggers on near-misses
- **Split Colab / local workflow** — sample generation runs on Colab (piper-tts has no WSL2 wheel); augmentation and training run locally on your GPU
- **WSL2 compatibility fixes** — soundfile-based audio decoding (replaces torchcodec), NumPy 2.0 shims, and RTX 5000 / Blackwell GPU workarounds
- **IPA phoneme input** — `"hˈeɪ fɹˈæŋk˺"` for consistent hard-K pronunciation across all TTS voices
- **ESPHome manifest output** — generates `hey_frank.json` ready to drop into your ESPHome config
- **Quality-of-life** — skip guards on all heavy steps, augmented clip preview cells, and tuned hyperparameters documented in the training history

### Repo contents

| File / Directory | Description |
|---|---|
| `hey_frank_training_notebook.ipynb` | Main training notebook |
| `atom-echo-s3r.yaml` | ESPHome config for M5Stack Atom Echo S3R |
| `models/` | Pre-trained hey_frank v3 and v4 `.tflite` models + ESPHome manifests |
| `microWakeWord/microwakeword/` | Three patched upstream files (see [Attribution](#attribution-and-authorship)) |

### Table of Contents

- [Environment](#environment)
- [Setup Instructions](#setup-instructions)
- [Quick Start](#quick-start)
- [Cell-by-Cell Guide](#cell-by-cell-guide)
- [Attribution and Authorship](#attribution-and-authorship)
- [Training History](#hey-frank--model-training-history)
- [License](#license--data-notice)

---

## Environment

This notebook is designed to run in two environments depending on the cell:

| Cells | Environment | Why |
|---|---|---|
| 1, 5–14 | **WSL2 on Windows** with an NVIDIA GPU | TensorFlow GPU support requires Linux; WSL2 gives you a Linux environment on Windows |
| 2–4 | **Google Colab** (or any Linux GPU) | `piper-tts` and `piper-phonemize` have no stable Windows/WSL2 wheel |

### WSL2 Setup Notes

- **WSL2** (Windows Subsystem for Linux 2) runs a real Linux kernel inside Windows. It has full CUDA/GPU passthrough for NVIDIA cards.
- Tested on: Windows 11, RTX 5070 Ti Laptop GPU (12 GB VRAM), Python 3.12 in a virtualenv
- You need: WSL2 with Ubuntu, CUDA-capable GPU driver, Python 3.12 venv with dependencies installed
- WSL2 default memory cap is 16 GB — if you have 32 GB RAM, set `memory=28GB` in `%USERPROFILE%\.wslconfig`

---

## Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/inventr-io/custom-micro-wake-word-model
cd custom-micro-wake-word-model
```

### 2. Clone the microWakeWord repo

Clone the upstream microWakeWord repo into the **same parent directory** as this repo:

```bash
cd ..
git clone https://github.com/kahrendt/microWakeWord
cd custom-micro-wake-word-model
```

Your directory structure should look like:
```
parent-dir/
├── custom-micro-wake-word-model/   ← this repo
└── microWakeWord/                  ← upstream repo
```

> **Note:** Cell 1 in the notebook also does this clone automatically if `microWakeWord/` doesn't already exist next to the notebook.

### 3. Apply the local modifications

This repo includes three patched files in `microWakeWord/microwakeword/`. From inside this repo, copy them over the upstream originals:

```bash
# From inside custom-micro-wake-word-model/:
cp microWakeWord/microwakeword/audio/clips.py  ../microWakeWord/microwakeword/audio/clips.py
cp microWakeWord/microwakeword/train.py        ../microWakeWord/microwakeword/train.py
cp microWakeWord/microwakeword/test.py         ../microWakeWord/microwakeword/test.py
```

See [Attribution and Authorship](#attribution-and-authorship) below for details on what each file fixes and why.

---

## Quick Start

1. Run **Cell 1** in Jupyter (in WSL2), restart the kernel
2. Run **Cells 2–4** in **Google Colab** (sample generation):
   - Cell 2: verify pronunciation
   - Cell 3: generate full positive sample set
   - Cell 4: generate confusable negative samples
3. Download `generated_samples/` and `confusable_negatives/` from Colab and place them in your WSL2 working directory
4. Run **Cells 5–12** in Jupyter (in WSL2) to download data and write the training config
5. Run **Cell 13** training via the **CLI** (recommended — see cell notes)
6. Run **Cell 14** to locate your `.tflite` and write the ESPHome manifest

---

## Cell-by-Cell Guide

### Cell 0 — Introduction / Quick-Edit Constants Reference

Markdown cell. Summarizes the key configurable parameters in one table so you don't have to hunt through each cell:

| Setting | Default | Notes |
|---|---|---|
| `target_word` | `"hˈeɪ fɹˈæŋk˺"` | IPA phoneme input — forces consistent hard K |
| `MAX_SAMPLES` | `50_000` | Reduce to `25_000` if disk space is tight |
| `PIPER_BATCH` | `256` | Reduce to `128` if you get CUDA OOM during sample generation |
| `CONFUSABLE_SAMPLES_PER_PHRASE` | `2000` | ~26k total across 13 phrases |
| `batch_size` (YAML) | `256` | Reduce to `128` if you get OOM during training |
| Training steps | `[20000, 15000, 10000]` | Use `[5000]` for a quick sanity-check run |
| `target_minimization` | `0.3` FA/hr | Lower = stricter false accept control |

---

### Cell 1 — Install microWakeWord *(run in WSL2)*

Clones the [microWakeWord](https://github.com/kahrendt/microWakeWord) repo and installs it as an editable package. Skips the clone if the directory already exists so re-runs are safe.

> **Tip:** You **must restart the kernel** after this cell before running anything else.

---

### Cell 2 — Wake Word Config + Single Preview Sample *(run in Google Colab)*

Sets up your wake word constants and generates **one sample WAV** so you can verify pronunciation before committing to a full run.

This notebook is configured to use `--phoneme-input` with the IPA string `"hˈeɪ fɹˈæŋk˺"` rather than plain text. Piper generates very short clips, and in short clips the final hard-K in "frank" can get dropped or softened by the TTS model. When that happens across thousands of samples, the model learns to trigger on "hey fran" nearly as readily as "hey frank" — producing a lot of false positives on similar phrases. The `˺` (no-audible-release) marker in the IPA string forces piper to consistently close the /k/, giving the model a clean signal to train on.

To use plain text instead, set `target_word = "hey frank"` and remove the `--phoneme-input` flag from the piper command. This works fine for many wake words, but be cautious with words that end in a hard stop consonant (k, t, p) — piper can drop or soften them in short clips, and that distortion will bake into your training data.

> **Tip:** Edit `target_word` at the top of this cell. If the preview audio sounds unclear or unnatural, try a phonetic spelling first (e.g. `'hey fraenk'`) to identify the issue, then translate to IPA.
>
> **Note:** This cell and Cells 3–4 should be run in Google Colab because `piper-tts` does not have a stable Windows/WSL2 wheel. Run them in Colab, then download the output folders and copy them to your WSL2 working directory.

---

### Cell 3 — Generate Full Training Sample Set *(run in Google Colab)*

Generates `MAX_SAMPLES` (default 50,000) TTS voice clips of your wake word using the libritts multi-speaker model. More speaker variety = more robust model.

Approximate times on a Colab T4:
- 10k samples: ~5 min
- 25k samples: ~12 min
- 50k samples: ~25 min

> **Tip:** Experiment with `--noise-scale` and `--noise-scale-w` flags (shown in cell comments) to vary pronunciation timing and style.
>
> **Note:** After this cell completes, download `generated_samples/` and copy it into your WSL2 working directory.

---

### Cell 4 — Generate Confusable Negative Samples *(run in Google Colab)*

Generates TTS clips of phonetically-similar phrases that must **not** trigger "hey frank" — e.g. `"hey fran"`, `"hey finn"`, `"frank"`, `"hey france"`. These become high-penalty hard negatives during training, teaching the model to discriminate fine-grained phonetic differences.

13 confusable phrases × `CONFUSABLE_SAMPLES_PER_PHRASE` (default 2,000) = ~26k total clips in `confusable_negatives/`.

> **Note:** Download `confusable_negatives/` from Colab and copy it to your WSL2 working directory alongside `generated_samples/`.

---

### Cell 5 — Download Augmentation Audio *(run in WSL2)*

Downloads three audio sources used for augmentation during training:
- **MIT Room Impulse Responses** — simulates rooms/reverb
- **AudioSet** — real-world ambient sounds and speech (~18k clips, streamed)
- **Free Music Archive (xsmall)** — music background noise

All downloads are skipped automatically if the files already exist.

> **Note:** Per the microWakeWord project, these datasets have mixed licenses. Any model trained with this data should be considered suitable for **non-commercial personal use only**.
>
> **Tip:** `audioset_16k/` and `fma_16k/` are each ~500 MB. This cell may take several minutes on first run.

---

### Cell 6 — Download Negative Datasets *(run in WSL2)*

Downloads pre-generated spectrogram features from the microWakeWord project for negative examples (things that are *not* the wake word):
- `speech` — general speech (hardest negatives)
- `dinner_party` — multi-speaker conversation + background noise
- `no_speech` — ambient sounds with no speech
- `dinner_party_eval` — held-out eval set for the FA/hr metric

Skip guard included — safe to re-run.

---

### Cell 7 — Configure Augmentation Pipeline *(run in WSL2)*

Defines how training clips are augmented before being fed to the model. Augmentation makes the model robust to real-world conditions.

Tuned settings vs. the original notebook:
- `remove_silence=True` — cleaner clips
- Wider jitter range (`0.10–0.50s`) — more positional variety
- Higher background SNR ceiling (`20 dB`) — includes softer backgrounds
- `GainTransition` added at `p=0.25` — simulates volume changes over time
- Stronger `AddBackgroundNoise` (`p=0.85`) and `RIR` (`p=0.60`)

> **Tip:** If the augmented preview (Cell 9) sounds completely unintelligible, raise `background_min_snr_db` toward `0` or lower `AddBackgroundNoise` probability.

---

### Cell 8 — Generate Spectrogram Features (Positive Samples) *(run in WSL2)*

Converts all augmented positive clips into 40-band spectrogram features — the actual format the model trains on. Generates three splits:

| Split | `slide_frames` | `repeat` | Purpose |
|---|---|---|---|
| training | 10 | 3 | 3× augmented versions per clip |
| validation | 10 | 1 | Single augmented version |
| testing | 1 | 1 | Simulates real streaming inference |

Existing splits are skipped automatically. If a partial mmap is detected it's cleaned up and regenerated.

> **Tip:** At 50k samples on a GPU this takes roughly 20–40 minutes. Don't close Jupyter during this step.

---

### Cell 9 — Preview Augmented Sample *(run in WSL2)*

Augments one random clip and plays it back. Run this a few times for variety. You should still be able to hear "hey frank" through the noise/reverb — muffled is fine, unintelligible is too much.

---

### Cell 10 — Generate Spectrogram Features (Confusable Negatives) *(run in WSL2)*

Runs the same augmentation and spectrogram pipeline on `confusable_negatives/`, outputting to `confusable_features/`. These are used as high-penalty negatives during training.

Idempotent — already-generated splits are skipped automatically.

---

### Cell 11 — Preview Augmented Confusable Sample *(run in WSL2)*

Spot-checks the confusable negative pipeline. You should hear something like "hey fran" or "hey finn" — clearly **not** "hey frank". If a clip sounds identical to the wake word, that phrase may need rephrasing.

---

### Cell 12 — Write Training Configuration YAML *(run in WSL2)*

Writes `training_parameters.yaml` with all training hyperparameters.

Key parameters to know:

| Parameter | Value | Notes |
|---|---|---|
| `training_steps` | `[25000, 20000]` | 45k total; use `[5000]` for a quick test |
| `negative_class_weight` | `[50, 60]` | **Most impactful knob for false accepts** — increase if too many FAs |
| `batch_size` | `256` | Tuned for 12 GB VRAM — reduce to `128` if OOM |
| `minimization_metric` | `ambient_false_positives_per_hour` | Trainer must beat `target_minimization` before maximizing recall |
| `target_minimization` | `0.3` | Acceptable false accepts per hour |

Batch composition (approximate share per training batch):

| Dataset | Weight | Notes |
|---|---|---|
| Positives (`generated_augmented_features`) | 6.0 | ~13% |
| `speech` | 10.0 | ~21% |
| `dinner_party` | 15.0 | ~32% — primary source of TV/ambient FAs |
| `no_speech` | 5.0 | ~11% |
| Confusables (`confusable_features`) | 8.0 | ~17% — high penalty (5.0) |

---

### Cell 13 — Train the Model *(run in WSL2)*

Launches the training run.

#### Recommended: run via CLI (not Jupyter)

For a long training run, the Jupyter kernel may time out or get killed by the OS. **It's more reliable to run training from the WSL2 command line.** The cell includes the CLI command as a comment block.

**Using `tmux` + `nohup` (recommended):**

```bash
# Start or attach to a persistent session
tmux new -s training
# or: tmux attach -t training

export XLA_FLAGS='--xla_gpu_autotune_level=0'
export PATH="$HOME/wakeword-env/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

nohup python3 -m microwakeword.model_train_eval \
  --training_config "training_parameters.yaml" \
  --train 1 --restore_checkpoint 0 \
  ... > training.log 2>&1 &

tail -f training.log

# Detach (leave running): Ctrl+B then D
# Reattach later: tmux attach -t training
```

> **Tip — Fresh start vs. resume:**
> - `--restore_checkpoint 0` — start fresh (use this if `trained_models/hey_frank/` doesn't exist yet)
> - `--restore_checkpoint 1` — resume from the last saved checkpoint
> - If you get `ValueError: model already exists`, delete `trained_models/hey_frank/` and re-run with `--restore_checkpoint 0`

**RTX 5000-series / Blackwell GPU notes:**
- The bundled `ptxas 12.9` is prepended to `$PATH` to override the system `ptxas 12.4` (which doesn't support Compute Capability 12.0)
- `XLA_FLAGS='--xla_gpu_autotune_level=0'` disables the XLA autotuner, which can fail on new GPU architectures

Watch the `estimated false positives per hour` metric at each 500-step validation checkpoint — you want it trending toward your `target_minimization` by the end of Phase 1.

---

### Cell 14 — Locate Model + Write ESPHome Manifest *(run in WSL2)*

After training completes, this cell:
1. Finds the quantized streaming `.tflite` model at `trained_models/hey_frank/tflite_stream_state_internal_quant/`
2. Reports the file size
3. Writes a starter `hey_frank.json` ESPHome manifest

> **Tip:** Set `probability_cutoff` in the manifest based on the cutoff table printed at the end of training. A value around `0.80–0.90` is typical — higher = more confident threshold required to trigger, so fewer false accepts but potentially lower recall.

Copy both `hey_frank.tflite` and `hey_frank.json` to your ESPHome configuration directory.

See: [ESPHome micro_wake_word docs](https://esphome.io/components/micro_wake_word) | [Model repo examples](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2)

---

## Attribution and Authorship

This notebook is based on [`basic_training_notebook.ipynb`](https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb) from the [microWakeWord](https://github.com/kahrendt/microWakeWord) project by [@kahrendt](https://github.com/kahrendt).

**I did not write the original code.** All credit for the microWakeWord framework, training pipeline, model architecture, and negative datasets belongs to the upstream authors.

What this notebook adds on top of the original:
- Wake word configured for `"hey frank"`
- Practical improvements for running on a **local Windows PC with an NVIDIA GPU via WSL2**
- Windows/Linux compatibility fixes (see below)
- Confusable negative sample generation pipeline
- Tuned hyperparameters for better false-accept behavior
- Quality-of-life additions: skip guards, error handling, ESPHome manifest output
- Updated addresses and download methods for augmentation resources and negative datasets where needed

### Local Modifications to microWakeWord

Three files have been patched from upstream and are included in this repo at `microWakeWord/microwakeword/`. See [Setup Instructions](#setup-instructions) above for how to apply them.

---

#### `microwakeword/audio/clips.py`
**Problem:** HuggingFace `datasets` uses `torchcodec` for audio decoding by default,
which has no Windows wheel and fails in WSL2.

**Fix:** Switched to `decode=False` + manual decoding via `soundfile` and `librosa`.
- Added `import io`, `import soundfile as sf`
- Changed `datasets.Audio()` → `datasets.Audio(decode=False)`
- Added `_decode_audio(audio_info, target_sr)` method that decodes manually,
  handles stereo→mono, and resamples if needed
- All `clip["audio"]["array"]` accesses replaced with `self._decode_audio(clip["audio"])`

> **Note:** This change is safe to run on Colab/Linux too — soundfile works everywhere.

---

#### `microwakeword/test.py`
**Problem:** NumPy 2.0 renamed `np.trapz` → `np.trapezoid`. Calling `np.trapz` raises
a deprecation warning or error depending on version.

**Fix:** Added compatibility shim:
```python
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
```

---

#### `microwakeword/train.py`
**Problem 1:** Same `np.trapz` → `np.trapezoid` rename as above.

**Fix:** Same compatibility shim applied in `validate_nonstreaming`.

**Problem 2:** `.numpy()` calls on metric values failed when values were plain
Python/NumPy scalars rather than TensorFlow tensors (version-dependent behavior).

**Fix:** Added `hasattr(x, "numpy")` guards before all `.numpy()` calls on
`fp`, `tp`, `fn` metric values in `validate_nonstreaming`.

---

## hey frank — Model Training History

### v2 — baseline (no confusables)

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[40, 50]` |
| `dinner_party` sampling / penalty | 10.0 / 1.5 |
| `speech` penalty | 2.0 |
| Confusable negatives | ❌ |
| Positive samples | Standard text input |
| `target_minimization` | 0.5 FA/hr |
| Training phases | `[30000, 20000]` |
| **Result** | Heavy false positives on similar phrases + ambient |

---

### v3 — confusable negatives added ✅

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[50, 60]` |
| `dinner_party` sampling / penalty | 10.0 / 2.0 |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ 13 phrases, sampling 8.0, penalty 5.0 |
| Positive samples | IPA phoneme input `hˈeɪ fɹˈæŋk˺` |
| `target_minimization` | 0.3 FA/hr |
| Training phases | `[20000, 15000, 10000]` |
| **Result** | **0.414 FA/hr, 97% recall — no confusable triggers, strong ambient robustness** |

---

### v4 — overcorrected ambient penalty

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[60, 75]` ← too aggressive |
| `dinner_party` sampling / penalty | 15.0 / 3.0 ← boosted |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ same as v3 |
| Positive samples | IPA phoneme input `hˈeɪ fɹˈæŋk˺` |
| `target_minimization` | 0.3 FA/hr |
| Training phases | `[25000, 20000]` |
| **Result** | 0.620 FA/hr, 94.5% recall — worse than v3 on both metrics |

---

### v5 — refined weights + expanded dataset ✅ deployed

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[50, 60]` ← reverted from v4 |
| `positive_class_weight` | `[2, 2]` ← raised from v3's `[1, 1]` |
| `dinner_party` sampling / penalty | 15.0 / 3.0 ← keeping v4 boost |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ 13 phrases, sampling 8.0, penalty 5.0 |
| Positive samples | IPA `hˈeɪ fɹˈæŋk˺`, sampling 8.0, penalty 2.0 |
| AudioSet clips | 18,683 (full balanced set) |
| `target_minimization` | 0.4 FA/hr ← relaxed from v3's 0.3 |
| Training phases | `[25000, 20000]` |
| **Result** | **0.103 FA/hr best min, 97.58% recall — best hey frank result** |

---

## Training History (hey_m5)

### hey_m5_v1 — initial model

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[40, 50]` |
| `positive_class_weight` | `[2, 2]` |
| `dinner_party` sampling / penalty | 15.0 / 3.0 |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ 5 phrases — hey em, hey five, em five, hey emma, hey emily |
| Confusable sampling / penalty | 8.0 / 5.0 |
| Positive samples | IPA `hˈeɪ \| ˈɛmfˈaɪv`, 50k TTS |
| `target_minimization` | 0.4 FA/hr |
| Training phases | `[25000, 20000]` |
| **Result** | **0.000 FA/hr @ 97.1% recall (cutoff 0.62 / uint8 158), 0.187 FA/hr @ 97.9% recall (cutoff 0.33 / uint8 84), 0.375 FA/hr @ 98.4% recall (cutoff 0.18 / uint8 46) — false triggers on "hey emma hi", "hey i'm tired"** |
| **ESPHome cutoffs** | Slightly sensitive = 84 (0.187 FA/hr), Moderately = 46 (0.375 FA/hr), Very = 26 |

---

### hey_m5_v2 — expanded confusables ⚠️ overcorrected

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[40, 50]` |
| `positive_class_weight` | `[2, 2]` |
| `dinner_party` sampling / penalty | 15.0 / 3.0 |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ 10 phrases — v1 set + hey emma hi, hey emily hi, hey i'm tired, hey i'm fired, i'm fired |
| Confusable sampling / penalty | 8.0 / 5.0 |
| Positive samples | IPA `hˈeɪ \| ˈɛmfˈaɪv`, 50k TTS |
| `target_minimization` | 0.4 FA/hr |
| Training phases | `[25000, 20000]` |
| **Result** | **0.187 FA/hr @ 92.7% recall (cutoff 0.58 / uint8 148) — ~5% recall loss vs v1 at every operating point; i'm tired/fired confusables too phonetically close to positive** |
| **ESPHome cutoffs** | Slightly sensitive = 148 (0.187 FA/hr), Moderately = 120 (0.375 FA/hr), Very = 64 (0.750 FA/hr) |

---

### hey_m5_v3 — trimmed confusables + real recordings

| Parameter | Value |
|---|---|
| `negative_class_weight` | `[40, 50]` |
| `positive_class_weight` | `[2, 2]` |
| `dinner_party` sampling / penalty | 15.0 / 3.0 |
| `speech` penalty | 2.5 |
| Confusable negatives | ✅ 8 phrases ← dropped hey i'm tired, hey i'm fired, hey emma hi; kept hey emily hi, hey i'm, i'm fired |
| Confusable sampling / penalty | 8.0 / 5.0 |
| Positive samples | IPA `hˈeɪ \| ˈɛmfˈaɪv`, 50k TTS + 217 real recordings |
| Real recordings sampling / penalty | 8.0 / 2.0 |
| `target_minimization` | 0.4 FA/hr |
| Training phases | `[25000, 20000]` |
| **Result** | **Best: 0.000 FA/hr @ 97.81% recall (validation). Test: 0.000 FA/hr @ 94.8% recall (cutoff 0.47 / uint8 120), 0.187 FA/hr @ 95.0% recall (cutoff 0.43 / uint8 110), 0.375 FA/hr @ 95.8% recall (cutoff 0.29 / uint8 74), 0.750 FA/hr @ 97.1% recall (cutoff 0.07 / uint8 18)** |
| **ESPHome cutoffs** | Slightly sensitive = 110 (0.187 FA/hr), Moderately = 74 (0.375 FA/hr), Very = 18 (0.750 FA/hr) |

---


## License / Data Notice

The negative training datasets (AudioSet, FMA, MIT RIRs) and the microWakeWord framework are subject to their respective licenses. Custom models trained with this data should be treated as suitable for **non-commercial personal use only**.

All framework code is by [@kahrendt](https://github.com/kahrendt) and contributors to [microWakeWord](https://github.com/kahrendt/microWakeWord).
