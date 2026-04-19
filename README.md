# micro-wake-word-operator

A custom [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) model for the wake word **"operator"**, trained for use with [ESPHome](https://esphome.io/components/micro_wake_word/) voice satellites.

Forked from [malonestar/custom-micro-wake-word-model](https://github.com/malonestar/custom-micro-wake-word-model) with a complete local training pipeline for Mac (Apple Silicon) via Docker.

---

## Using the Model

Add to your ESPHome config:

```yaml
micro_wake_word:
  models:
    - model: github://erinnichols/micro-wake-word-operator/models/operator.json@main
      id: ethel
```

Both `operator.json` and `operator.tflite` are included in this repository.
**Note:** `operator` is a reserved keyword in C++ / esphome, so I named mine ethel.

### Model stats

| Metric | Value |
|--------|-------|
| Wake word | "operator" |
| Best FA/hr | 0.103 |
| Recall | 96.5% |
| Training samples | 50,000 (synthetic TTS) |
| Model size | ~61 KB |
| Architecture | MixedNet |

---

## Training Your Own Model

This repo includes a complete, reproducible training pipeline that runs locally on Mac via Docker — no GPU required, no Colab timeouts.

### Prerequisites

- Docker Desktop (tested on Mac Apple Silicon, linux/amd64 emulation)
- ~20GB free disk space
- A HuggingFace account and [access token](https://huggingface.co/settings/tokens) (free, needed for dataset downloads)
- Amphetamine or `caffeinate` to prevent Mac sleep during long runs

### Quick start

```bash
git clone https://github.com/erinnichols/micro-wake-word-operator
cd micro-wake-word-operator
docker compose build
docker compose run microwakeword-trainer python3 train_operator.py
```

Training takes approximately 6-8 hours on Apple Silicon (CPU emulation). The script is fully idempotent — if it crashes, just rerun and it picks up where it left off.

### What the script does

1. Downloads the LibriTTS-R piper TTS model
2. Generates 50,000 synthetic "operator" voice clips across 904 speakers
3. Generates confusable negative samples (opera, operate, liberator, etc.)
4. Downloads augmentation data (MIT RIRs, AudioSet, FMA)
5. Downloads pre-built negative datasets (speech, dinner party, ambient)
6. Generates spectrogram features for all datasets
7. Trains a MixedNet model for 45,000 steps in two phases
8. Exports `operator.tflite` and `operator.json`

### Compatibility fixes included

The training pipeline includes several patches for running on modern Python/numpy environments that the upstream notebook doesn't address:

- **PyTorch 2.6**: `torch.load()` `weights_only` default change — patched in `piper-sample-generator`
- **numpy 2.x**: `np.trapz` removed → `np.trapezoid`; `.numpy()` calls on numpy arrays — patched in `microwakeword` source
- **torchaudio**: `set_audio_backend` removed in newer versions — no-op patched
- **microwakeword pip package**: Missing `audio` submodule — installs from OHF-Voice source instead
- **Docker memory**: Validation spikes to ~28GB; requires Docker Desktop memory limit ≥ 28GB

### Confusable phrases

The model was trained to reject these phonetically similar phrases:

```python
confusable_phrases = [
    "operate",
    "opera", 
    "operative",
    "operators",
    "operate her",
    "liberator",
    "decorator",
    "alligator",
    "narrate her",
    "operate now",
]
```

---

## Hardware context

This model was developed for use with the [Waveshare ESP32-S3 AI Smart Speaker](https://www.waveshare.com/esp32-s3-audio-board.htm) running ESPHome, as part of a Home Assistant voice satellite fleet. The satellites use the [sw3Dan/waveshare-s2-audio_esphome_voice](https://github.com/sw3Dan/waveshare-s2-audio_esphome_voice) ESPHome configuration.

The wake word "operator" is styled after a 1960s telephone switchboard operator persona.

---

## Credits

- Upstream notebook: [malonestar/custom-micro-wake-word-model](https://github.com/malonestar/custom-micro-wake-word-model)
- microWakeWord framework: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)
- TTS sample generation: [rhasspy/piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
- ESPHome microWakeWord docs: [esphome.io/components/micro_wake_word](https://esphome.io/components/micro_wake_word/)

---

## License

Model trained on data with mixed licenses. Suitable for **non-commercial personal use only** per the microWakeWord project's dataset licensing terms.
