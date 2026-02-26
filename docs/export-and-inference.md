# Export & Inference

The export stage converts the trained PyTorch classifier to ONNX for deployment. The inference engine runs all three ONNX stages in a streaming fashion for real-time wake word detection.

**Source:** `src/livewakeword/export/onnx.py`, `src/livewakeword/inference/engine.py`
**CLI:** `livewakeword export <config>`, `livewakeword detect <model>`

## ONNX Export

### Classifier Export

`export_classifier()` exports the trained PyTorch classifier head to ONNX format.

| Property | Value |
|----------|-------|
| Input name | `embeddings` |
| Input shape | `(1, 16, 96)` with dynamic batch axis |
| Output name | `score` |
| Output shape | `(1, 1)` with dynamic batch axis |
| Opset version | 13 |

### Full Pipeline Export

`export_full_pipeline()` assembles all three ONNX models into one directory:

```
output/<model_name>/
├── <model_name>.onnx          # Trained classifier head
├── melspectrogram.onnx         # Frozen mel-spectrogram frontend
└── embedding_model.onnx        # Frozen speech-embedding CNN
```

The mel-spectrogram and speech-embedding models are copied from `data/models/` (where `livewakeword setup` placed them). They cannot be fused into a single ONNX graph because they are separate pre-trained models.

### INT8 Quantization

`quantize_onnx()` applies dynamic INT8 quantization using `onnxruntime.quantization`:

- Weight type: `QuantType.QInt8`
- Output filename: `<model_name>.int8.onnx`

Enable via the `--quantize` flag:

```bash
livewakeword export configs/hey_jarvis.yaml --quantize
```

### Export Entry Point

`run_export()` loads the trained model from `output/<model_name>/<model_name>.pt`, exports it to ONNX, and optionally quantizes it. Raises `FileNotFoundError` if the trained model doesn't exist.

## Streaming Inference Engine

**Source:** `src/livewakeword/inference/engine.py`

`StreamingWakeWordEngine` processes audio in 80ms frames through the full ONNX pipeline.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE` | 16,000 | Audio sample rate |
| `FRAME_MS` | 80 | Frame duration in milliseconds |
| `FRAME_SAMPLES` | 1,280 | Samples per frame (16000 * 80 / 1000) |

### Initialization

```python
StreamingWakeWordEngine(
    classifier_path: str | Path,    # Path to ONNX classifier
    models_dir: str | Path,         # Directory with mel + embedding ONNX models
    threshold: float = 0.5,         # Detection threshold
    cooldown_ms: float = 2000.0     # Minimum time between detections (ms)
)
```

### Internal Buffers

| Buffer | Type | Description |
|--------|------|-------------|
| `_mel_buffer` | `list[np.ndarray]` | Accumulated mel-spectrogram frames |
| `_embedding_buffer` | `deque(maxlen=16)` | Last 16 embeddings (fixed-size ring buffer) |
| `_frames_since_detection` | `int` | Cooldown counter |

### Frame Processing

`process_frame(audio_frame)` processes a single 80ms audio chunk:

```
80ms audio (1280 samples)
    │
    ▼
1. Convert int16 → float32 (÷ 32768)
    │
    ▼
2. MelSpectrogramFrontend (ONNX)
   → ~5 mel frames appended to buffer
    │
    ▼
3. Check: enough mel frames? (need >= 76)
   No → return 0.0
    │
    ▼
4. Extract latest 76-frame window
    │
    ▼
5. SpeechEmbedding (ONNX)
   → 1 embedding appended to ring buffer
    │
    ▼
6. Trim mel buffer (cap at 76 + 8×20 frames)
    │
    ▼
7. Check: 16 embeddings in buffer?
   No → return 0.0
    │
    ▼
8. Stack 16 embeddings → (1, 16, 96)
    │
    ▼
9. Run classifier (ONNX)
   → score [0, 1]
    │
    ▼
10. Apply cooldown logic
    │
    ▼
Return score
```

### Cooldown Logic

After a detection (score >= threshold), further detections are suppressed for `cooldown_ms` milliseconds (default: 2000ms). This prevents repeated triggers for a single utterance.

The cooldown counter (`_frames_since_detection`) increments each frame and resets to 0 on detection. Detections are only valid when the counter has reached `cooldown_frames = cooldown_ms / FRAME_MS`.

### Detection API

Two methods are available:

| Method | Returns | Description |
|--------|---------|-------------|
| `process_frame(audio)` | `float` | Raw score [0, 1] |
| `detect(audio)` | `bool` | `True` if score >= threshold and cooldown expired |

### Buffer Management

The mel buffer is trimmed to prevent unbounded memory growth. After each embedding extraction, the buffer is capped at `76 + 8 * 20 = 236` mel frames, keeping enough history for the sliding window while bounding memory usage.

### Reset

`reset()` clears all buffers and resets the cooldown counter, allowing the engine to be reused from a clean state.

## Real-Time Detection

`run_detect()` provides a ready-to-use microphone detection loop.

### Usage

```bash
livewakeword detect output/hey_jarvis/hey_jarvis.onnx \
    --models-dir ./data/models \
    --threshold 0.5
```

### Audio Capture

Uses PyAudio to capture audio from the default microphone:

| Parameter | Value |
|-----------|-------|
| Format | int16 (paInt16) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Buffer size | 1,280 samples (80ms) |

### Detection Loop

1. Open PyAudio stream with the above parameters
2. Read 1,280 samples per iteration
3. Convert bytes to int16 numpy array
4. Pass to `engine.detect(frame)`
5. Log detection events
6. Continue until `KeyboardInterrupt`
7. Clean up stream and PyAudio instance
