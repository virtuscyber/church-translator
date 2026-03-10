"""Voice Activity Detection (VAD) based audio chunking.

Instead of fixed-duration chunks, this module uses energy-based VAD to detect
speech pauses and splits audio on natural sentence boundaries. This produces
more coherent transcriptions and better translations.

Strategy:
- Buffer incoming audio frames
- Use RMS energy + zero-crossing rate to classify frames as speech/silence
- When we detect a pause (silence after speech), emit the chunk
- Enforce min/max chunk durations to avoid too-short or too-long segments
- On max duration, find the best split point (longest recent silence gap)
"""

from __future__ import annotations

import io
import logging
import wave
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Analysis frame size
FRAME_MS = 30
FRAME_RATE = 16000
FRAME_SAMPLES = int(FRAME_RATE * FRAME_MS / 1000)  # 480 samples


class EnergyVAD:
    """Simple but effective energy + zero-crossing based VAD.
    
    Adapts thresholds over time to handle varying room noise.
    More robust than webrtcvad for church environments (reverb, music).
    """
    
    def __init__(self, aggressiveness: int = 2):
        # Higher aggressiveness = higher threshold = more filtering
        self._energy_threshold_base = [0.005, 0.008, 0.012, 0.018][min(aggressiveness, 3)]
        self._energy_floor = self._energy_threshold_base
        self._energy_ceiling = self._energy_threshold_base * 10
        self._noise_estimate = self._energy_threshold_base
        self._adaptation_rate = 0.05
        self._frame_count = 0
    
    def is_speech(self, audio_float32: np.ndarray) -> bool:
        """Determine if an audio frame contains speech."""
        rms = np.sqrt(np.mean(audio_float32 ** 2))
        
        # Adaptive noise floor
        self._frame_count += 1
        if rms < self._noise_estimate * 1.5:
            # Likely noise — slowly adapt
            self._noise_estimate = (
                (1 - self._adaptation_rate) * self._noise_estimate 
                + self._adaptation_rate * rms
            )
        
        threshold = max(self._energy_floor, self._noise_estimate * 3.0)
        threshold = min(threshold, self._energy_ceiling)
        
        return rms > threshold


class VADChunker:
    """Splits an audio stream into speech-bounded chunks using energy VAD.
    
    Args:
        aggressiveness: VAD aggressiveness (0-3). Higher = more aggressive
            at filtering non-speech. 2-3 recommended for noisy church audio.
        min_chunk_sec: Minimum chunk duration before we'll emit.
        max_chunk_sec: Maximum chunk duration — force-emit even without a pause.
        silence_threshold_sec: How long silence must last to trigger a split.
        padding_sec: Extra audio to include before/after speech boundaries.
        input_sample_rate: Sample rate of incoming audio.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        min_chunk_sec: float = 3.0,
        max_chunk_sec: float = 15.0,
        silence_threshold_sec: float = 0.8,
        padding_sec: float = 0.3,
        input_sample_rate: int = 48000,
        enable_preview: bool = False,
        preview_after_sec: float = 2.0,
    ):
        self.vad = EnergyVAD(aggressiveness)
        
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        self.silence_threshold_sec = silence_threshold_sec
        self.padding_sec = padding_sec
        self.input_sample_rate = input_sample_rate
        self.enable_preview = enable_preview
        self.preview_after_sec = preview_after_sec
        
        # Frame size in input sample rate
        self._frame_samples = int(input_sample_rate * FRAME_MS / 1000)
        
        # Internal state
        self._audio_buffer: list[np.ndarray] = []
        self._buffered_seconds: float = 0.0
        self._speech_started = False
        self._silence_frames = 0
        self._speech_frames_total = 0
        
        # Track silence gaps for smart force-splitting
        # Each entry: (buffer_seconds_at_gap, gap_duration_frames)
        self._silence_gaps: deque = deque(maxlen=50)
        self._current_gap_start: Optional[float] = None
        
        # Preview tracking — emit early snapshot for speculative STT
        self._preview_emitted = False
        
        # Leftover audio from last feed that didn't fill a frame
        self._leftover = np.array([], dtype=np.float32)

    def feed(self, audio: np.ndarray) -> list[tuple[str, bytes]]:
        """Feed audio data and get back any completed chunks.
        
        Returns:
            List of (tag, wav_bytes) tuples where tag is:
            - "preview": Early snapshot for speculative STT (more audio coming)
            - "final": Complete chunk at speech boundary
        """
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        
        self._audio_buffer.append(audio.copy())
        self._buffered_seconds += len(audio) / self.input_sample_rate
        
        # Prepend leftover from last call
        if len(self._leftover) > 0:
            analysis_audio = np.concatenate([self._leftover, audio])
        else:
            analysis_audio = audio
        
        chunks_out: list[tuple[str, bytes]] = []
        
        offset = 0
        while offset + self._frame_samples <= len(analysis_audio):
            frame = analysis_audio[offset:offset + self._frame_samples]
            offset += self._frame_samples
            
            is_speech = self.vad.is_speech(frame)
            
            if is_speech:
                # Track end of silence gap
                if self._silence_frames > 0 and self._current_gap_start is not None:
                    gap_duration = self._silence_frames * FRAME_MS / 1000.0
                    if gap_duration >= 0.15:  # Only track meaningful gaps (>150ms)
                        self._silence_gaps.append((self._current_gap_start, gap_duration))
                    self._current_gap_start = None
                
                self._silence_frames = 0
                self._speech_frames_total += 1
                if not self._speech_started:
                    self._speech_started = True
            else:
                if self._silence_frames == 0:
                    self._current_gap_start = self._buffered_seconds
                self._silence_frames += 1
            
            silence_duration = self._silence_frames * FRAME_MS / 1000.0
            
            if self._speech_started:
                # Natural pause boundary — ideal split point
                if (silence_duration >= self.silence_threshold_sec 
                        and self._buffered_seconds >= self.min_chunk_sec):
                    chunk = self._emit_chunk()
                    if chunk:
                        chunks_out.append(("final", chunk))
                
                # Force-split at max duration — find best split point
                elif self._buffered_seconds >= self.max_chunk_sec:
                    chunk = self._emit_chunk_smart()
                    if chunk:
                        chunks_out.append(("final", chunk))
                
                # Preview emission — early snapshot for speculative STT
                elif (self.enable_preview 
                      and not self._preview_emitted
                      and self._buffered_seconds >= self.preview_after_sec
                      and self._speech_frames_total > 0):
                    preview = self._snapshot_preview()
                    if preview:
                        chunks_out.append(("preview", preview))
                        self._preview_emitted = True
        
        # Save leftover
        self._leftover = analysis_audio[offset:]
        
        return chunks_out

    def flush(self) -> Optional[bytes]:
        """Flush any remaining buffered audio as a final chunk."""
        if self._audio_buffer and self._buffered_seconds >= 0.5:
            return self._emit_chunk()
        return None

    def _snapshot_preview(self) -> Optional[bytes]:
        """Create a WAV snapshot of the current buffer WITHOUT resetting state.
        
        This allows speculative STT to begin while audio continues accumulating.
        """
        if not self._audio_buffer:
            return None
        
        all_audio = np.concatenate(self._audio_buffer, axis=0)
        duration = len(all_audio) / self.input_sample_rate
        
        logger.info("VAD preview: %.1fs snapshot for speculative STT", duration)
        return self._to_wav(all_audio)

    def _emit_chunk_smart(self) -> Optional[bytes]:
        """Force-emit at max duration but split at the best silence gap.
        
        Instead of cutting mid-word at exactly max_chunk_sec, look back
        through recent silence gaps and split at the longest one (most
        likely a sentence boundary). Remainder stays in buffer for the
        next chunk, preserving continuity.
        """
        if not self._audio_buffer or not self._silence_gaps:
            # No gaps tracked — fall back to regular emit
            return self._emit_chunk()
        
        # Find the longest silence gap in the back half of the buffer
        # (prefer splitting later to keep chunks substantial)
        half_point = self._buffered_seconds * 0.4  # Only consider gaps after 40%
        best_gap = None
        best_duration = 0.0
        
        for gap_time, gap_dur in self._silence_gaps:
            if gap_time >= half_point and gap_dur > best_duration:
                best_gap = gap_time
                best_duration = gap_dur
        
        if best_gap is None:
            return self._emit_chunk()
        
        # Split at the best gap
        all_audio = np.concatenate(self._audio_buffer, axis=0)
        split_sample = int(best_gap * self.input_sample_rate)
        
        # Add padding after the split point
        padding_samples = int(self.padding_sec * self.input_sample_rate)
        split_sample = min(split_sample + padding_samples, len(all_audio))
        
        chunk_audio = all_audio[:split_sample]
        remainder = all_audio[split_sample:]
        
        duration = len(chunk_audio) / self.input_sample_rate
        logger.info(
            "VAD smart split: %.1fs chunk (split at %.1fs silence gap of %.2fs, %.1fs remainder)",
            duration, best_gap, best_duration, len(remainder) / self.input_sample_rate,
        )
        
        # Reset state, keeping remainder in buffer
        self._audio_buffer = [remainder] if len(remainder) > 0 else []
        self._buffered_seconds = len(remainder) / self.input_sample_rate
        self._speech_started = len(remainder) > 0
        self._silence_frames = 0
        self._speech_frames_total = 0
        self._silence_gaps.clear()
        self._current_gap_start = None
        self._preview_emitted = False
        self._leftover = np.array([], dtype=np.float32)
        
        return self._to_wav(chunk_audio)

    def _emit_chunk(self) -> Optional[bytes]:
        """Concatenate buffered audio and return as WAV bytes, then reset."""
        if not self._audio_buffer:
            return None
        
        all_audio = np.concatenate(self._audio_buffer, axis=0)
        
        # Trim trailing silence (keep a little padding)
        padding_samples = int(self.padding_sec * self.input_sample_rate)
        speech_end = len(all_audio)
        
        if len(all_audio) > padding_samples * 2:
            window = int(0.05 * self.input_sample_rate)
            for i in range(len(all_audio) - window, padding_samples, -window):
                rms = np.sqrt(np.mean(all_audio[i:i+window] ** 2))
                if rms > 0.01:
                    speech_end = min(i + padding_samples, len(all_audio))
                    break
        
        chunk_audio = all_audio[:speech_end]
        duration = len(chunk_audio) / self.input_sample_rate
        
        logger.info(
            "VAD chunk: %.1fs (speech frames: %d, silence: %.1fs)",
            duration,
            self._speech_frames_total,
            self._silence_frames * FRAME_MS / 1000.0,
        )
        
        # Reset state
        self._audio_buffer = []
        self._buffered_seconds = 0.0
        self._speech_started = False
        self._silence_frames = 0
        self._speech_frames_total = 0
        self._silence_gaps.clear()
        self._current_gap_start = None
        self._preview_emitted = False
        self._leftover = np.array([], dtype=np.float32)
        
        return self._to_wav(chunk_audio)

    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert float32 mono audio to WAV bytes."""
        pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.input_sample_rate)
            wf.writeframes(pcm_int16.tobytes())
        return buf.getvalue()


class FileVADChunker:
    """Convenience wrapper to chunk a WAV file using VAD.
    
    Used by the test script as an alternative to fixed-duration splitting.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        min_chunk_sec: float = 3.0,
        max_chunk_sec: float = 15.0,
        silence_threshold_sec: float = 0.8,
    ):
        self.aggressiveness = aggressiveness
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        self.silence_threshold_sec = silence_threshold_sec

    def chunk_file(self, wav_path: str) -> list[bytes]:
        """Read a WAV file and split into VAD-bounded chunks.
        
        Args:
            wav_path: Path to mono 16-bit WAV file (any sample rate).
            
        Returns:
            List of WAV byte buffers, one per speech segment.
        """
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        
        # Convert to float32 mono
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]
        
        chunker = VADChunker(
            aggressiveness=self.aggressiveness,
            min_chunk_sec=self.min_chunk_sec,
            max_chunk_sec=self.max_chunk_sec,
            silence_threshold_sec=self.silence_threshold_sec,
            input_sample_rate=sample_rate,
        )
        
        # Feed in 100ms blocks to simulate streaming
        # Preview is disabled for file chunking — only emit finals
        block_samples = int(sample_rate * 0.1)
        all_chunks: list[bytes] = []
        
        for start in range(0, len(audio), block_samples):
            block = audio[start:start + block_samples]
            tagged_chunks = chunker.feed(block)
            for tag, wav_bytes in tagged_chunks:
                if tag == "final":
                    all_chunks.append(wav_bytes)
        
        # Flush remaining
        final = chunker.flush()
        if final:
            all_chunks.append(final)
        
        total_duration = len(audio) / sample_rate
        logger.info(
            "VAD split %s: %d chunks from %.1fs of audio (avg %.1fs/chunk)",
            wav_path, len(all_chunks), total_duration,
            total_duration / max(len(all_chunks), 1),
        )
        
        return all_chunks
