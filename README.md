# livewakeword

Wake word detection with frozen ONNX feature extraction and trainable PyTorch classifiers.

livewakeword generates synthetic training data via TTS, augments it with real-world noise, extracts features through pre-trained ONNX models, and trains a lightweight classifier head — all from a single YAML config.

## Installation

```bash
uv sync                    # Core dependencies
uv sync --group train      # Training (TTS, augmentation, librosa, scikit-learn)
uv sync --group export     # ONNX export (onnx, onnxscript)
uv sync --group detect     # Real-time microphone detection
uv sync --group dev        # Development (pytest, ruff, mypy)
uv sync --all-extras       # Everything
```

Requires Python 3.11+.

**System dependency:** `espeak-ng` is required for TTS phonemization:
```bash
brew install espeak-ng     # macOS
sudo apt install espeak-ng # Ubuntu/Debian
```

## Quick Start

### 1. Download pre-trained ONNX models and data

```bash
livewakeword setup
```

Downloads mel-spectrogram and speech-embedding ONNX models, VITS TTS checkpoint, background noise, room impulse responses, and ACAV100M features.

### 2. Create a config

```yaml
# configs/hey_jarvis.yaml
model_name: hey_jarvis
target_phrases:
  - "hey jarvis"

n_samples: 10000
n_samples_val: 2000

model:
  model_type: dnn
  model_size: small

steps: 50000
learning_rate: 0.0001
max_negative_weight: 1500
target_fp_per_hour: 0.2
```

See [`configs/hey_jarvis.yaml`](configs/hey_jarvis.yaml) for a complete example.

### 3. Run the full pipeline

```bash
livewakeword run configs/hey_jarvis.yaml
```

Or run each stage individually:

```bash
livewakeword generate configs/hey_jarvis.yaml   # TTS synthesis + adversarial negatives
livewakeword augment  configs/hey_jarvis.yaml   # Augment audio + extract features
livewakeword train    configs/hey_jarvis.yaml   # 3-phase adaptive training
livewakeword export   configs/hey_jarvis.yaml   # Export classifier to ONNX
```

### 4. Real-time detection

```bash
livewakeword detect output/hey_jarvis/hey_jarvis.onnx
```

## Architecture

```
Raw audio (16kHz)
    |
    v
MelSpectrogramFrontend (ONNX)        n_fft=512, hop=160, n_mels=32
    |  (batch, time_frames, 32)
    v
SpeechEmbedding (ONNX)               76-frame window, stride=8
    |  (batch, n_windows, 96)         Google speech_embedding CNN, 96-dim output
    v
Classifier (PyTorch)                  DNN (FC+LayerNorm) or RNN (Bi-LSTM)
    |  (batch, 1)                     Input: last 16 timesteps → (batch, 16, 96)
    v
Detection score [0, 1]
```

The two ONNX feature extraction stages are frozen and shared across all wake words. Only the classifier head is trained, making training fast and data-efficient.

## CLI Reference

| Command | Description |
|---------|-------------|
| `livewakeword setup` | Download frozen ONNX models, VITS TTS checkpoint, background noise, and RIRs |
| `livewakeword generate <config>` | Synthesize positive and adversarial negative clips via VITS with SLERP speaker blending |
| `livewakeword augment <config>` | Augment clips (pitch, EQ, RIR, backgrounds) and extract features to `.npy` |
| `livewakeword train <config>` | 3-phase adaptive training with hard example mining |
| `livewakeword export <config>` | Export trained classifier to ONNX (optional `--quantize` for INT8) |
| `livewakeword run <config>` | Full pipeline: generate → augment → train → export |
| `livewakeword detect <model>` | Real-time wake word detection from microphone |

## Model Sizes

| Size | Hidden Dim | Blocks | Architecture |
|------|-----------|--------|--------------|
| tiny | 16 | 1 | FC + LayerNorm or Bi-LSTM |
| small | 32 | 1 | FC + LayerNorm or Bi-LSTM |
| medium | 128 | 2 | FC + LayerNorm or Bi-LSTM |
| large | 256 | 3 | FC + LayerNorm or Bi-LSTM |

Configure via `model.model_type` (`dnn` or `rnn`) and `model.model_size` in your YAML config.

## Config Reference

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | *required* | Name for the model and output directory |
| `target_phrases` | *required* | Wake word phrases to detect |
| `n_samples` | 10000 | Training samples per class |
| `n_samples_val` | 2000 | Validation samples per class |
| `tts_batch_size` | 50 | VITS inference batch size |
| `custom_negative_phrases` | `[]` | Additional negative phrases beyond auto-generated |
| `noise_scales` | `[0.98]` | Overall speech variability |
| `noise_scale_ws` | `[0.98]` | Phoneme duration variability |
| `length_scales` | `[0.75, 1.0, 1.25]` | Speaking rate multipliers (slow/normal/fast) |
| `slerp_weights` | `[0.5]` | Speaker interpolation weights (0=speaker1, 1=speaker2) |
| `max_speakers` | `null` | Cap on speaker IDs from the model (null = all 904) |
| `data_dir` | `./data` | Root data directory |
| `output_dir` | `./output` | Root output directory |
| `augmentation.rounds` | 1 | Number of augmentation passes |
| `augmentation.batch_size` | 16 | Augmentation batch size |
| `augmentation.background_paths` | `[./data/backgrounds]` | Background noise directories |
| `augmentation.rir_paths` | `[./data/rirs]` | Room impulse response directories |
| `model.model_type` | `dnn` | Classifier type: `dnn` or `rnn` |
| `model.model_size` | `small` | Model size: `tiny`, `small`, `medium`, `large` |
| `steps` | 50000 | Training steps (phase 1) |
| `learning_rate` | 0.0001 | Base learning rate |
| `max_negative_weight` | 1500 | Max BCE weight for negative class |
| `target_fp_per_hour` | 0.2 | Target false positives per hour |

## Documentation

Detailed documentation for each pipeline stage:

- [Architecture Overview](docs/overview.md)
- [Data Generation](docs/data-generation.md) — TTS synthesis and adversarial negatives
- [Augmentation](docs/augmentation.md) — Audio augmentation and alignment
- [Feature Extraction](docs/feature-extraction.md) — Mel spectrogram and speech embeddings
- [Training](docs/training.md) — 3-phase adaptive training
- [Export & Inference](docs/export-and-inference.md) — ONNX export and real-time detection

## License

MIT
