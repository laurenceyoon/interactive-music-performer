import librosa
import numpy as np
import pyaudio
import queue
from typing import Optional
import time

from ..config import CHUNK_SIZE, HOP_LENGTH, SAMPLE_RATE, CHANNELS, N_FFT


class StreamProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, verbose=False):
        self.chunk_size = chunk_size
        self.channels = CHANNELS
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.format = pyaudio.paFloat32
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.buffer = queue.Queue()
        self.chroma_buffer = queue.Queue()
        self.last_chunk = None
        self.is_mic_open = False
        self.frame_index = 0
        self.audio_y = np.array([])

    def _process_frame(self, data, frame_count, time_into, status_flag):
        self.buffer.put(data)
        if self.verbose:
            print(f"[ARRIVED] {self.frame_index}st frame: {time.time()}")

        query_audio = np.frombuffer(data, dtype=np.float32)
        self.audio_y = np.concatenate((self.audio_y, query_audio)) if self.audio_y.any() else query_audio
        query_chroma_stft = librosa.feature.chroma_stft(
            y=query_audio, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        if self.last_chunk is None:  # first audio chunk is given
            self.chroma_buffer.put(query_chroma_stft[:, :-1])  # pop last frame converted with zero padding
        else:
            override_previous_padding = librosa.feature.chroma_stft(
                y=np.concatenate((self.last_chunk, query_audio[:HOP_LENGTH])),
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
            )[:, 1:-1]  # drop first and last frame converted with zero padding
            accumulated_chroma = np.concatenate((override_previous_padding, query_chroma_stft[:, 1:-1]), axis=1)
            self.chroma_buffer.put(accumulated_chroma)
        
        self.last_chunk = query_audio[query_audio.shape[0] - HOP_LENGTH:]
        self.frame_index += 1
        return (data, pyaudio.paContinue)

    def run(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._process_frame,
        )
        self.is_mic_open = True
        self.audio_stream.start_stream()
        print("* Recording in progress....")

    def stop(self):
        if self.is_mic_open:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.is_mic_open = False
            self.audio_interface.terminate()
            print("Recording Stopped.")


sp = StreamProcessor()
