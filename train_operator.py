#!/usr/bin/env python3
"""
Train a custom microWakeWord model for the wake word "operator".
Run inside the microwakeword-trainer Docker container.

Usage:
    docker compose run microwakeword-trainer python3 train_operator_mww.py

Steps:
    1. Generate positive TTS samples
    2. Generate confusable negative samples
    3. Download augmentation data (RIRs, AudioSet, FMA)
    4. Download pre-built negative datasets
    5. Generate spectrogram features
    6. Train model
    7. Export .tflite + .json manifest
"""

import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_WORD = "operator"
MODEL_NAME = "operator"
MAX_SAMPLES = 50_000
PIPER_BATCH = 64          # lower for CPU emulation; raise to 256 if you have a GPU
CONFUSABLE_SAMPLES_PER_PHRASE = 1000

MODEL_PATH = "models/en_US-libritts_r-medium.pt"
MODEL_CONFIG_PATH = f"{MODEL_PATH}.json"

CONFUSABLE_PHRASES = [
    'operate',
    'opera',
    'operative',
    'operators',
    'operate her',
    'liberator',
    'decorator',
    'alligator',
    'narrate her',
    'operate now',
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def run(cmd, **kwargs):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)

def wget(url, output):
    print(f"Downloading {url} -> {output}")
    urllib.request.urlretrieve(url, output)

def run_piper(args):
    """Run piper_sample_generator, streaming output in real time."""
    result = subprocess.run(
        [sys.executable, "-m", "piper_sample_generator", *args],
        text=True,
    )
    result.check_returncode()

def count_wavs(directory):
    return len(list(Path(directory).glob("*.wav")))

# ── Step 0: Verify environment ─────────────────────────────────────────────────
print("\n=== Verifying environment ===")
print(f"Working directory: {os.getcwd()}")
try:
    import microwakeword
    import tensorflow as tf
    print(f"tensorflow: {tf.__version__}")
    print(f"microwakeword: OK")
except ImportError as e:
    print(f"ERROR: missing dependency: {e}")
    sys.exit(1)

# ── Step 1: Download TTS model ─────────────────────────────────────────────────
print("\n=== Step 1: TTS model ===")
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading LibriTTS-R model (~300MB)...")
    wget(
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
        MODEL_PATH,
    )

if not os.path.exists(MODEL_CONFIG_PATH):
    print("Downloading model config...")
    wget(
        "https://raw.githubusercontent.com/rhasspy/piper-sample-generator/master/models/en_US-libritts_r-medium.pt.json",
        MODEL_CONFIG_PATH,
    )
print("TTS model ready")

# ── Step 2: Generate positive samples ─────────────────────────────────────────
print(f"\n=== Step 2: Generate {MAX_SAMPLES} positive samples ===")
os.makedirs("generated_samples", exist_ok=True)
existing = count_wavs("generated_samples")
if existing >= MAX_SAMPLES:
    print(f"Already have {existing} samples, skipping")
else:
    print(f"Have {existing}, generating up to {MAX_SAMPLES}...")
    run_piper([
        TARGET_WORD,
        "--model", MODEL_PATH,
        "--max-samples", str(MAX_SAMPLES),
        "--batch-size", str(PIPER_BATCH),
        "--noise-scales", "0.5",
        "--noise-scale-ws", "0.6",
        "--output-dir", "generated_samples",
    ])
    print(f"Generated {count_wavs('generated_samples')} samples")

# ── Step 3: Generate confusable negative samples ───────────────────────────────
print(f"\n=== Step 3: Generate confusable negatives ===")
os.makedirs("confusable_negatives", exist_ok=True)
for phrase in CONFUSABLE_PHRASES:
    safe_name = phrase.replace(" ", "_")
    existing = len(list(Path("confusable_negatives").glob(f"{safe_name}_*.wav")))
    if existing >= CONFUSABLE_SAMPLES_PER_PHRASE:
        print(f'  "{phrase}": {existing} samples already present, skipping')
        continue
    print(f'  "{phrase}" -> {CONFUSABLE_SAMPLES_PER_PHRASE} samples...')
    tmp_dir = f"/tmp/confusable_{safe_name}"
    os.makedirs(tmp_dir, exist_ok=True)
    run_piper([
        phrase,
        "--model", MODEL_PATH,
        "--max-samples", str(CONFUSABLE_SAMPLES_PER_PHRASE),
        "--batch-size", str(PIPER_BATCH),
        "--output-dir", tmp_dir,
    ])
    copied = 0
    for wav in sorted(Path(tmp_dir).glob("*.wav")):
        dest = Path("confusable_negatives") / f"{safe_name}_{wav.name}"
        shutil.copy(wav, dest)
        copied += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    {copied} samples -> confusable_negatives/")
total = len(list(Path("confusable_negatives").glob("*.wav")))
print(f"{total} total confusable negative samples")

# ── Step 4: Download augmentation audio ───────────────────────────────────────
print("\n=== Step 4: Download augmentation audio ===")
import datasets as hf_datasets
import fsspec
import librosa
import soundfile as sf
from tqdm import tqdm

def decode_audio(audio_info, target_sr=16000):
    data = audio_info.get("bytes")
    if not data:
        path = audio_info.get("path") or ""
        if path.startswith("hf://"):
            with fsspec.open(path, "rb") as f:
                data = f.read()
        else:
            data = open(path, "rb").read()
    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr

# MIT Room Impulse Responses
rir_wavs = list(Path("mit_rirs").glob("*.wav")) if Path("mit_rirs").exists() else []
if len(rir_wavs) < 10:
    print(f"Downloading MIT RIRs (found {len(rir_wavs)} existing)...")
    os.makedirs("./mit_rirs", exist_ok=True)
    rir_ds = hf_datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
        features=hf_datasets.Features({"audio": hf_datasets.Audio(decode=False)}),
    )
    for i, row in enumerate(tqdm(rir_ds)):
        raw_name = (row["audio"].get("path") or "").split("/")[-1].split("\\")[-1]
        name = raw_name if raw_name.lower().endswith(".wav") else f"rir_{i:04d}.wav"
        try:
            arr = decode_audio(row["audio"])
            scipy.io.wavfile.write(f"mit_rirs/{name}", 16000, (arr * 32767).astype(np.int16))
        except Exception as e:
            print(f"  Skipped row {i}: {e}")
    print(f"MIT RIRs done ({len(list(Path('mit_rirs').glob('*.wav')))} files)")
else:
    print(f"MIT RIRs already present ({len(rir_wavs)} files)")

# AudioSet
AUDIOSET_CLIPS = 18683
if not os.path.exists("audioset_16k") or len(list(Path("audioset_16k").glob("*.wav"))) < 100:
    print(f"Downloading AudioSet ({AUDIOSET_CLIPS} clips)...")
    os.makedirs("audioset_16k", exist_ok=True)
    ds = hf_datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True, trust_remote_code=True
    )
    ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
    for i, row in enumerate(tqdm(ds, total=AUDIOSET_CLIPS)):
        if i >= AUDIOSET_CLIPS:
            break
        try:
            arr = decode_audio(row["audio"])
            scipy.io.wavfile.write(f"audioset_16k/{i:05d}.wav", 16000, (arr * 32767).astype(np.int16))
        except Exception:
            pass
    print("AudioSet done")
else:
    print("AudioSet already present")

# Free Music Archive
if not os.path.exists("fma"):
    print("Downloading FMA xsmall...")
    os.makedirs("fma", exist_ok=True)
    wget(
        "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
        "fma/fma_xs.zip",
    )
    with zipfile.ZipFile("fma/fma_xs.zip") as zf:
        zf.extractall("fma")

if not os.path.exists("fma_16k") or len(list(Path("fma_16k").glob("*.wav"))) < 100:
    print("Converting FMA to 16kHz WAV...")
    os.makedirs("fma_16k", exist_ok=True)
    for p in tqdm(sorted(Path("fma").glob("**/*.mp3"))):
        try:
            arr, _ = librosa.load(str(p), sr=16000, mono=True)
            scipy.io.wavfile.write(f"fma_16k/{p.stem}.wav", 16000, (arr * 32767).astype(np.int16))
        except Exception:
            pass
    print("FMA done")
else:
    print("FMA already present")

# ── Step 5: Download pre-built negative datasets ───────────────────────────────
print("\n=== Step 5: Download negative datasets ===")
if not os.path.exists("./negative_datasets"):
    os.makedirs("./negative_datasets", exist_ok=True)
    link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    for fname in ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]:
        zip_path = f"negative_datasets/{fname}"
        wget(link_root + fname, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("negative_datasets")
        print(f"  {fname} done")
    print("All negative datasets ready")
else:
    print("Negative datasets already present")

# ── Step 6: Generate spectrogram features ─────────────────────────────────────
print("\n=== Step 6: Generate spectrogram features ===")
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

clips = Clips(
    input_directory="generated_samples",
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=True,
    random_split_seed=42,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.15,
        "TanhDistortion":        0.10,
        "PitchShift":            0.15,
        "BandStopFilter":        0.10,
        "AddColorNoise":         0.20,
        "AddBackgroundNoise":    0.85,
        "Gain":                  1.00,
        "GainTransition":        0.25,
        "RIR":                   0.60,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k", "audioset_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=20,
    min_jitter_s=0.10,
    max_jitter_s=0.50,
)

os.makedirs("generated_augmented_features", exist_ok=True)
split_config = {
    "training":   {"split_name": "train",      "repetition": 3, "slide_frames": 10},
    "validation": {"split_name": "validation", "repetition": 1, "slide_frames": 10},
    "testing":    {"split_name": "test",       "repetition": 1, "slide_frames": 1},
}
for split, cfg in split_config.items():
    out_dir = f"generated_augmented_features/{split}"
    mmap_path = f"{out_dir}/wakeword_mmap"
    if os.path.exists(mmap_path) and list(os.scandir(mmap_path)):
        print(f"  {split}: already exists, skipping")
        continue
    if os.path.exists(mmap_path):
        shutil.rmtree(mmap_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Generating {split}...")
    try:
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=cfg["slide_frames"],
            step_ms=10,
        )
        RaggedMmap.from_generator(
            out_dir=mmap_path,
            sample_generator=spectrograms.spectrogram_generator(
                split=cfg["split_name"],
                repeat=cfg["repetition"],
            ),
            batch_size=200,
            verbose=True,
        )
        print(f"  {split} done")
    except Exception:
        traceback.print_exc()
        if os.path.exists(mmap_path):
            shutil.rmtree(mmap_path)
        sys.exit(1)

# Confusable features
confusable_clips = Clips(
    input_directory="confusable_negatives",
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=True,
    random_split_seed=99,
    split_count=0.1,
)
os.makedirs("confusable_features", exist_ok=True)
confusable_split_config = {
    "training":   {"split_name": "train",      "repetition": 2, "slide_frames": 10},
    "validation": {"split_name": "validation", "repetition": 1, "slide_frames": 10},
    "testing":    {"split_name": "test",       "repetition": 1, "slide_frames": 1},
}
for split, cfg in confusable_split_config.items():
    out_dir = f"confusable_features/{split}"
    mmap_path = f"{out_dir}/wakeword_mmap"
    if os.path.exists(mmap_path) and list(os.scandir(mmap_path)):
        print(f"  confusable {split}: already exists, skipping")
        continue
    if os.path.exists(mmap_path):
        shutil.rmtree(mmap_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Generating confusable {split}...")
    try:
        spectrograms = SpectrogramGeneration(
            clips=confusable_clips,
            augmenter=augmenter,
            slide_frames=cfg["slide_frames"],
            step_ms=10,
        )
        RaggedMmap.from_generator(
            out_dir=mmap_path,
            sample_generator=spectrograms.spectrogram_generator(
                split=cfg["split_name"],
                repeat=cfg["repetition"],
            ),
            batch_size=200,
            verbose=True,
        )
        print(f"  confusable {split} done")
    except Exception:
        traceback.print_exc()
        if os.path.exists(mmap_path):
            shutil.rmtree(mmap_path)
        sys.exit(1)

# ── Step 7: Write training config ─────────────────────────────────────────────
print("\n=== Step 7: Writing training config ===")
config = {
    "window_step_ms": 10,
    "train_dir": f"trained_models/{MODEL_NAME}",
    "features": [
        {
            "features_dir":        "generated_augmented_features",
            "sampling_weight":     8.0,
            "penalty_weight":      2.0,
            "truth":               True,
            "truncation_strategy": "truncate_start",
            "type":                "mmap",
        },
        {
            "features_dir":        "negative_datasets/speech",
            "sampling_weight":     10.0,
            "penalty_weight":      2.5,
            "truth":               False,
            "truncation_strategy": "random",
            "type":                "mmap",
        },
        {
            "features_dir":        "negative_datasets/dinner_party",
            "sampling_weight":     15.0,
            "penalty_weight":      3.0,
            "truth":               False,
            "truncation_strategy": "random",
            "type":                "mmap",
        },
        {
            "features_dir":        "negative_datasets/no_speech",
            "sampling_weight":     5.0,
            "penalty_weight":      1.0,
            "truth":               False,
            "truncation_strategy": "random",
            "type":                "mmap",
        },
        {
            "features_dir":        "negative_datasets/dinner_party_eval",
            "sampling_weight":     0.0,
            "penalty_weight":      1.0,
            "truth":               False,
            "truncation_strategy": "split",
            "type":                "mmap",
        },
        {
            "features_dir":        "confusable_features",
            "sampling_weight":     8.0,
            "penalty_weight":      5.0,
            "truth":               False,
            "truncation_strategy": "random",
            "type":                "mmap",
        },
    ],
    "training_steps":        [25000, 20000],
    "positive_class_weight": [2,     2],
    "negative_class_weight": [40,    50],
    "learning_rates":        [0.001, 0.0001],
    "batch_size":            256,
    "time_mask_max_size":    [5, 5],
    "time_mask_count":       [1, 1],
    "freq_mask_max_size":    [3, 3],
    "freq_mask_count":       [1, 1],
    "eval_step_interval":    500,
    "clip_duration_ms":      1500,
    "target_minimization":   0.4,
    "minimization_metric":   "ambient_false_positives_per_hour",
    "maximization_metric":   "average_viable_recall",
}

os.makedirs(f"trained_models/{MODEL_NAME}", exist_ok=True)
with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f)
print("training_parameters.yaml written")

# ── Step 8: Train ──────────────────────────────────────────────────────────────
print("\n=== Step 8: Training model (this will take a while!) ===")

# Patch numpy 2.x compatibility in microwakeword train.py
print("Patching microwakeword train.py for numpy 2.x compatibility...")
train_py = "/tmp/microwakeword-src/microwakeword/train.py"
with open(train_py) as f:
    src = f.read()
src = src.replace('result["fp"].numpy()', 'np.array(result["fp"])')
src = src.replace('result["tp"].numpy()', 'np.array(result["tp"])')
src = src.replace('ambient_predictions["tp"].numpy()', 'np.array(ambient_predictions["tp"])')
src = src.replace('ambient_predictions["fp"].numpy()', 'np.array(ambient_predictions["fp"])')
src = src.replace('ambient_predictions["fn"].numpy()', 'np.array(ambient_predictions["fn"])')
src = src.replace('np.trapz(', 'np.trapezoid(')
with open(train_py, "w") as f:
    f.write(src)
print("Patched!")

test_py = "/tmp/microwakeword-src/microwakeword/test.py"
with open(test_py) as f:
    src = f.read()
src = src.replace('np.trapz(', 'np.trapezoid(')
with open(test_py, "w") as f:
    f.write(src)
print("Patched test.py!")

subprocess.run([
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", "training_parameters.yaml",
    "--train", "0",
    "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0",
    "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1",
    "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
], check=True)

# ── Step 9: Export model ───────────────────────────────────────────────────────
print("\n=== Step 9: Exporting model ===")
tflite_src = f"trained_models/{MODEL_NAME}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
tflite_dest = f"{MODEL_NAME}.tflite"
json_dest = f"{MODEL_NAME}.json"

if os.path.exists(tflite_src):
    size_kb = os.path.getsize(tflite_src) / 1024
    print(f"Model size: {size_kb:.1f} KB")

    manifest = {
        "type": "micro",
        "wake_word": TARGET_WORD,
        "author": "erin",
        "website": "",
        "model": tflite_dest,
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": 0.50,
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        }
    }

    with open(json_dest, "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.copy2(tflite_src, tflite_dest)

    print(f"\nDone! Output files:")
    print(f"  {tflite_dest}")
    print(f"  {json_dest}")
    print(f"\nAdd to your ESPHome yaml:")
    print(f"""
micro_wake_word:
  models:
    - model: /config/esphome/{json_dest}
      id: operator
""")
else:
    print(f"ERROR: Model not found at {tflite_src}")
    print("Check that training completed successfully")
    sys.exit(1)
