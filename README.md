# Custom micro-wake-word models in ESPHome

A Jupyter notebook for training a custom wake word model for **"hey frank"** using the [microWakeWord](https://github.com/kahrendt/microWakeWord) framework, deployable to **M5Stack and other ESP32 devices via ESPHome**.

Also a working ESPHome YAML configuration file for the M5Stack Atom Echo S3R, demonstrating how to upload and use your custom model on-device.  

[<img src="/media/device_setup_example.jpg" width="400" alt="image of Homeassistant voice assistant setup" />](/media/device_setup_example.jpg)
---

## Attribution and Authorship

This notebook is based on [`basic_training_notebook.ipynb`](https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb) from the [microWakeWord](https://github.com/kahrendt/microWakeWord) project by [@kahrendt](https://github.com/kahrendt).

**I did not write the original code.** All credit for the microWakeWord framework, training pipeline, model architecture, and negative datasets belongs to the upstream authors.

What this notebook adds on top of the original:
- Wake word configured for `"hey frank"`
- Practical improvements for running on a **local Windows PC with an NVIDIA GPU via WSL2**
- Windows/Linux compatibility fixes (see below)
- Tuned hyperparameters for better false-accept behavior
- Quality-of-life additions: skip guards, error handling, ESPHome manifest output
- Updated addresses and download methods for augmentation resources and negative dataset where needed

---

## Environment

This notebook is designed to run in two environments depending on the cell:

| Cells | Environment | Why |
|---|---|---|
| 1, 4–11 | **WSL2 on Windows** with an NVIDIA GPU | TensorFlow GPU support requires Linux; WSL2 gives you a Linux environment on Windows |
| 2–3 | **Google Colab** (or any Linux GPU) | `piper-phonemize` has no Windows wheel and can be unstable in WSL2 |

### WSL2 Setup Notes

- **WSL2** (Windows Subsystem for Linux 2) runs a real Linux kernel inside Windows. It has full CUDA/GPU passthrough for NVIDIA cards.
- Tested on: Windows 11, RTX 5070 Ti Laptop GPU (12 GB VRAM), Python 3.12 in a virtualenv
- You need: WSL2 with Ubuntu, CUDA-capable GPU driver, Python 3.12 venv with dependencies installed
- WSL2 default memory cap is 16 GB — if you have 32 GB RAM, set `memory=28GB` in `%USERPROFILE%\.wslconfig`

---

## Quick Start

1. Run **Cell 1** in Jupyter (in WSL2), restart the kernel
2. Run **Cells 2–3** in **Google Colab** (sample generation) — download the `generated_samples/` folder when done
3. Place `generated_samples/` in your notebook working directory on WSL2
4. Run **Cells 4–9** in Jupyter (in WSL2) to prepare data and config
5. Run **Cell 10** training via the **CLI** (recommended — see cell notes)
6. Run **Cell 11** to locate your `.tflite` and write the ESPHome manifest

---

## Cell-by-Cell Guide

### Cell 0 — Introduction / Quick-Edit Constants Reference

Markdown cell. Summarizes the key configurable parameters in one table so you don't have to hunt through each cell:

| Setting | Default | Tip |
|---|---|---|
| `target_word` | `'hey frank'` | Use phonetic spelling if TTS pronunciation sounds wrong |
| `MAX_SAMPLES` | `50_000` | Reduce to `25_000` if disk space is tight |
| `PIPER_BATCH` | `256` | Reduce to `128` if you get CUDA OOM during sample generation |
| `batch_size` (YAML) | `256` | Reduce to `128` if you get OOM during training |
| Training steps | `[30000, 20000]` | Use `[5000]` for a quick sanity-check run |
| `target_minimization` | `2.0` FA/hr | Lower (e.g. `0.5`) for a quieter, stricter model |

---

### Cell 1 — Install microWakeWord

Clones the [microWakeWord](https://github.com/kahrendt/microWakeWord) repo and installs it as an editable package. Skips the clone if the directory already exists so re-runs are safe.

> **Tip:** You **must restart the kernel** after this cell before running anything else.

---

### Cell 2 — Wake Word Config + Single Preview Sample *(run in Google Colab)*

Sets up your wake word constants and generates **one sample WAV** so you can verify pronunciation before committing to a full run.

> **Tip:** Edit `target_word` at the top of this cell. If the preview audio sounds unclear or unnatural, try a phonetic spelling (e.g. `'hey fraenk'` or `'hey frahnk'`).
>
> **Note:** This cell and Cell 3 were run in Google Colab for this example, because `piper-phonemize` does not have a stable Windows/WSL2 wheel. Run them in Colab, then download the `generated_samples/` folder and copy it to your WSL2 working directory before running Cell 4.

---

### Cell 3 — Generate Full Training Sample Set *(run in Google Colab)*

Generates `MAX_SAMPLES` (default 50,000) TTS voice clips of your wake word using the libritts multi-speaker model. More speaker variety = more robust model.

Approximate times on a Colab T4:
- 10k samples: ~5 min
- 25k samples: ~12 min
- 50k samples: ~25 min

> **Tip:** Experiment with `--noise-scale` and `--noise-scale-w` flags (shown in cell comments) to vary pronunciation timing and style.
>
> **Note:** After this cell completes, download the `generated_samples/` folder from Colab and copy it into the `notebooks/` directory on your local machine / WSL2 path before proceeding.

---

### Cell 4 — Download Augmentation Audio

Downloads three audio sources used for augmentation during training:
- **MIT Room Impulse Responses** — simulates rooms/reverb
- **AudioSet** — real-world ambient sounds and speech (2000 clips, streamed)
- **Free Music Archive (xsmall)** — music background noise

All downloads are skipped automatically if the files already exist.

> **Note:** Per the microWakeWord project, these datasets have mixed licenses. Any model trained with this data should be considered suitable for **non-commercial personal use only**.
>
> **Tip:** `audioset_16k/` and `fma_16k/` are each ~500 MB. This cell may take several minutes on first run.

---

### Cell 5 — Configure Augmentation Pipeline

Defines how training clips are augmented before being fed to the model. Augmentation makes the model robust to real-world conditions.

Tuned settings vs. the original notebook:
- `remove_silence=True` — cleaner clips
- Wider jitter range (`0.10–0.50s`) — more positional variety
- Higher background SNR ceiling (`20 dB`) — includes softer backgrounds
- `GainTransition` added at `p=0.25` — simulates volume changes over time
- Stronger `AddBackgroundNoise` (`p=0.85`) and `RIR` (`p=0.60`)

> **Tip:** If the augmented preview (Cell 6) sounds completely unintelligible, raise `background_min_snr_db` toward `0` or lower `AddBackgroundNoise` probability.

---

### Cell 6 — Preview Augmented Sample

Augments one random clip and plays it back. Run this a few times for variety. You should still be able to hear "hey frank" through the noise/reverb — muffled is fine, unintelligible is too much.

---

### Cell 7 — Generate Spectrogram Features

Converts all augmented clips into 40-band spectrogram features — the actual format the model trains on. Generates three splits:

| Split | `slide_frames` | `repeat` | Purpose |
|---|---|---|---|
| training | 10 | 3 | 3× augmented versions per clip |
| validation | 10 | 1 | Single augmented version |
| testing | 1 | 1 | Simulates real streaming inference |

Existing splits are skipped automatically. If a partial mmap is detected it's cleaned up and regenerated.

> **Tip:** At 50k samples on a GPU this takes roughly 20–40 minutes. Don't close Jupyter during this step.

---

### Cell 8 — Download Negative Datasets

Downloads pre-generated spectrogram features from the microWakeWord project for negative examples (things that are *not* the wake word):
- `speech` — general speech (hardest negatives)
- `dinner_party` — multi-speaker conversation + background noise
- `no_speech` — ambient sounds with no speech
- `dinner_party_eval` — held-out eval set for the FA/hr metric

Skip guard included — safe to re-run.

---

### Cell 9 — Write Training Configuration YAML

Writes `training_parameters.yaml` with all training hyperparameters.

Key parameters to know:

| Parameter | Value | Notes |
|---|---|---|
| `training_steps` | `[30000, 20000]` | 50k total; use `[5000]` for a quick test |
| `negative_class_weight` | `[25, 30]` | **Most impactful knob for false accepts** — increase toward 40–50 if too many FAs |
| `batch_size` | `256` | Tuned for 12 GB VRAM — reduce to 128 if OOM |
| `minimization_metric` | `ambient_false_positives_per_hour` | The model first tries to get under `target_minimization` FA/hr |
| `target_minimization` | `2.0` | Acceptable false accepts per hour — lower for stricter behavior |

---

### Cell 10 — Train the Model

Launches the training run.

#### Recommended: run via CLI (not Jupyter)

For a 50k-step training run (~2–4 hours), the Jupyter kernel may time out or get killed by the OS. **It's more reliable to run training from the WSL2 command line.** The cell includes the CLI command as a comment block.

**Using `tmux` + `nohup` (recommended):**

```bash
# Start or attach to a persistent session
tmux new -s training
# or: tmux attach -t training

# Then run (the full nohup command is in the cell comments):
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

**RTX 5000-series / Blackwell GPU notes (included in cell):**
- The bundled `ptxas 12.9` is prepended to `$PATH` to override the system `ptxas 12.4` (which doesn't support Compute Capability 12.0)
- `XLA_FLAGS='--xla_gpu_autotune_level=0'` is set to avoid XLA autotuner failures on new GPU architectures

Watch the `estimated false positives per hour` metric at each 500-step validation checkpoint — you want it trending toward `2.0` or below by the end of Phase 1.

---

### Cell 11 — Locate Model + Write ESPHome Manifest

After training completes, this cell:
1. Finds the quantized streaming `.tflite` model at `trained_models/hey_frank/tflite_stream_state_internal_quant/`
2. Reports the file size
3. Writes a starter `hey_frank.json` ESPHome manifest

> **Tip:** Set `probability_cutoff` in the manifest based on the cutoff table printed at the end of training. A value around `0.80–0.90` is typical — higher = more confident threshold required to trigger, so fewer false accepts but potentially lower recall.

Copy both `hey_frank.tflite` and `hey_frank.json` to your ESPHome configuration directory.

See: [ESPHome micro_wake_word docs](https://esphome.io/components/micro_wake_word) | [Model repo examples](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2)

---

## License / Data Notice

The negative training datasets (AudioSet, FMA, MIT RIRs) and the microWakeWord framework are subject to their respective licenses. Custom models trained with this data should be treated as suitable for **non-commercial personal use only**.

All framework code is by [@kahrendt](https://github.com/kahrendt) and contributors to [microWakeWord](https://github.com/kahrendt/microWakeWord).
