"""Silero VAD wrapper using ONNX runtime directly (no torchaudio dependency)."""

import numpy as np
import onnxruntime

SILERO_ONNX_PATH = (
    "/home/user/.cache/torch/hub/snakers4_silero-vad_master"
    "/src/silero_vad/data/silero_vad.onnx"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # 32ms at 16kHz — required by Silero


class SileroVAD:
    """Streaming Silero VAD that processes 512-sample chunks at 16kHz."""

    def __init__(
        self,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_speech_ms: int = 250,
        min_silence_ms: int = 800,
        pre_speech_pad_ms: int = 300,
    ):
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
        self.min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
        self.min_silence_samples = int(SAMPLE_RATE * min_silence_ms / 1000)
        self.pre_speech_pad_samples = int(SAMPLE_RATE * pre_speech_pad_ms / 1000)

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            SILERO_ONNX_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        self._context_size = 64  # 16kHz context
        self.reset_states()

    def reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._speech_start_sample = 0

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Process a 512-sample float32 chunk. Returns speech probability [0, 1]."""
        if chunk.shape[0] != WINDOW_SIZE:
            raise ValueError(f"Expected {WINDOW_SIZE} samples, got {chunk.shape[0]}")

        chunk = chunk.astype(np.float32).reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        out, state = self.session.run(None, ort_inputs)

        self._state = state
        self._context = x[:, -self._context_size:]

        return float(out[0][0])

    def __call__(self, chunk: np.ndarray) -> dict | None:
        """Process chunk and return speech boundary events.

        Returns:
            {"start": sample} when speech starts
            {"end": sample} when speech ends (after min_silence_ms of silence)
            None otherwise
        """
        self._current_sample += WINDOW_SIZE
        prob = self.process_chunk(chunk)

        # Speech resumes after tentative end — cancel the end
        if prob >= self.threshold and self._temp_end:
            self._temp_end = 0

        # Speech start
        if prob >= self.threshold and not self._triggered:
            self._triggered = True
            self._speech_start_sample = self._current_sample
            start = max(0, self._current_sample - self.pre_speech_pad_samples - WINDOW_SIZE)
            return {"start": start}

        # Below negative threshold while in speech — potential end
        if prob < self.neg_threshold and self._triggered:
            if not self._temp_end:
                self._temp_end = self._current_sample

            silence_duration = self._current_sample - self._temp_end
            if silence_duration >= self.min_silence_samples:
                speech_duration = self._temp_end - self._speech_start_sample
                self._temp_end = 0
                self._triggered = False

                # Only emit end if speech was long enough
                if speech_duration >= self.min_speech_samples:
                    return {"end": self._temp_end or self._current_sample}

                # Too short — reset and ignore
                return None

        return None

    @property
    def is_speaking(self) -> bool:
        return self._triggered
