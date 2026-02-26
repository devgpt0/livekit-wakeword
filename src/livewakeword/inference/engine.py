"""Real-time streaming wake word detection engine."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from livewakeword.models.feature_extractor import (
    EMBEDDING_ONNX_FILENAME,
    MEL_ONNX_FILENAME,
    MelSpectrogramFrontend,
    SpeechEmbedding,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 80  # 80ms frames for real-time processing
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 1280 samples per frame


class StreamingWakeWordEngine:
    """Streaming wake word detection using ONNX runtime for all stages.

    Processes 80ms audio frames through:
    mel-spectrogram (ONNX) → speech embedding (ONNX) → classifier (ONNX).
    """

    def __init__(
        self,
        classifier_path: str | Path,
        models_dir: str | Path,
        threshold: float = 0.5,
        cooldown_ms: float = 2000.0,
    ):
        """Initialize streaming engine.

        Args:
            classifier_path: Path to ONNX classifier model
            models_dir: Directory containing melspectrogram.onnx and embedding_model.onnx
            threshold: Detection threshold (0-1)
            cooldown_ms: Minimum time between detections in milliseconds
        """
        import onnxruntime as ort

        self.threshold = threshold
        self.cooldown_frames = int(cooldown_ms / FRAME_MS)

        models_dir = Path(models_dir)

        # Load all three ONNX models
        self._mel_frontend = MelSpectrogramFrontend(
            onnx_path=models_dir / MEL_ONNX_FILENAME,
        )
        self._speech_embedding = SpeechEmbedding(
            onnx_path=models_dir / EMBEDDING_ONNX_FILENAME,
        )
        self._classifier_session = ort.InferenceSession(
            str(classifier_path),
            providers=["CPUExecutionProvider"],
        )
        self._classifier_input_name = self._classifier_session.get_inputs()[0].name

        # Audio buffer: accumulate raw audio and run mel on the full buffer
        # so STFT frames match full-clip extraction regardless of mel model
        # internals (n_fft, hop, centering mode).
        self._max_audio_seconds = 3.0  # enough for 16 embeddings at any mel config
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frame_count = 0  # mel frames already seen

        self._embedding_buffer: deque[np.ndarray] = deque(maxlen=16)
        self._frames_since_detection = self.cooldown_frames  # Allow immediate detection
        self._mel_frames_since_embedding = 0  # Track mel frames for stride matching

    def process_frame(self, audio_frame: np.ndarray) -> float:
        """Process a single 80ms audio frame.

        Args:
            audio_frame: (1280,) int16 or float32 audio samples at 16kHz

        Returns:
            Detection score (0-1). Above threshold indicates wake word detected.
        """
        # Convert int16 to float32 if needed
        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0

        self._frames_since_detection += 1

        # Accumulate raw audio and compute mel on the full buffer so that
        # STFT frames are identical to full-clip extraction.
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_frame])

        all_mel = self._mel_frontend(self._audio_buffer)
        if all_mel.ndim == 3:
            all_mel = all_mel[0]  # (total_frames, 32)

        new_mel_frames = all_mel.shape[0] - self._mel_frame_count
        self._mel_frame_count = all_mel.shape[0]
        self._mel_frames_since_embedding += new_mel_frames

        # Need at least 76 mel frames for one embedding window
        if all_mel.shape[0] < 76:
            return 0.0

        # Only extract a new embedding every 8 mel frames (matching training stride)
        if self._mel_frames_since_embedding < 8:
            return self._last_score if hasattr(self, "_last_score") else 0.0

        # Extract embedding from the latest 76-frame window
        self._mel_frames_since_embedding = 0
        window = all_mel[-76:]  # (76, 32)
        embedding = self._speech_embedding(window[np.newaxis, :, :])  # (1, 96)
        self._embedding_buffer.append(embedding[0])

        # Trim audio buffer to avoid unbounded growth (time-based, model-agnostic)
        max_audio_samples = int(self._max_audio_seconds * SAMPLE_RATE)
        if len(self._audio_buffer) > max_audio_samples:
            self._audio_buffer = self._audio_buffer[-max_audio_samples:]
            # Recount mel frames from the trimmed buffer
            trimmed_mel = self._mel_frontend(self._audio_buffer)
            if trimmed_mel.ndim == 3:
                trimmed_mel = trimmed_mel[0]
            self._mel_frame_count = trimmed_mel.shape[0]

        # Need 16 embeddings for classifier
        if len(self._embedding_buffer) < 16:
            return 0.0

        # Run classifier
        emb_sequence = np.stack(list(self._embedding_buffer), axis=0)
        emb_input = emb_sequence[np.newaxis, :, :].astype(np.float32)  # (1, 16, 96)

        outputs = self._classifier_session.run(None, {self._classifier_input_name: emb_input})
        score = float(outputs[0][0, 0])
        self._last_score = score

        # Apply cooldown
        if score >= self.threshold and self._frames_since_detection >= self.cooldown_frames:
            self._frames_since_detection = 0

        return score

    def detect(self, audio_frame: np.ndarray) -> bool:
        """Process frame and return whether wake word was detected.

        Applies threshold and cooldown logic.
        """
        score = self.process_frame(audio_frame)
        return score >= self.threshold and self._frames_since_detection == 0

    def reset(self) -> None:
        """Reset internal buffers."""
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frame_count = 0
        self._embedding_buffer.clear()
        self._frames_since_detection = self.cooldown_frames
        self._mel_frames_since_embedding = 0
        self._last_score = 0.0


def run_detect(
    classifier_path: str,
    models_dir: str = "./data/models",
    threshold: float = 0.5,
) -> None:
    """Run real-time detection from microphone.

    Requires pyaudio and onnxruntime.
    """
    import pyaudio

    engine = StreamingWakeWordEngine(
        classifier_path=classifier_path,
        models_dir=models_dir,
        threshold=threshold,
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )

    logger.info(f"Listening for wake word (threshold={threshold})... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16)
            detected = engine.detect(frame)
            score = engine._last_score if hasattr(engine, "_last_score") else 0.0
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " DETECTED!" if detected else ""
            print(f"\r  [{bar}] {score:.3f}{marker}    ", end="", flush=True)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
