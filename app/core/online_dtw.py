import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional

from ..config import HOP_LENGTH, CHUNK_SIZE, Direction, N_FFT
from ..models import Schedule
from .stream_processor import StreamProcessor

matplotlib.use("agg")

measures = [
    0.0,
    0.5,
    2.0,
    3.5,
    5.0,
    6.5,
    8.0,
    9.5,
    11.0,
    12.5,
    14.0,
    15.0,
    17.0,
    18.5,
    20.0,
    21.5,
    23.0,
    24.5,
    26.0,
]
from dtw import dtw


class OnlineTimeWarping:
    def __init__(
        self,
        sp: StreamProcessor,
        ref_audio_path,
        window_size,
        max_run_count=30,
        hop_length=HOP_LENGTH,
        verbose=False,
    ):
        self.sp = sp
        self.window_size = window_size
        self.max_run_count = max_run_count
        self.hop_length = hop_length
        self.verbose = verbose
        self.query_pointer = 0
        self.ref_pointer = 0
        self.time_length = 0
        self.distance = 0
        self.run_count = 0
        self.previous_direction = None
        self.current_query_stft = None  # (12, N)
        self.query_stft = np.array([])  # (12, n)
        self.query_audio = np.array([])
        self.index1s = np.array([])
        self.index2s = np.array([])
        self.warping_path = None
        self.warping_path_time = None
        self.cost_matrix = None
        self.iteration = 0

        self.initialize_ref_audio(ref_audio_path)

    def initialize_ref_audio(self, audio_path):
        audio_y, sr = librosa.load(audio_path)
        self.ref_audio = audio_y
        self.ref_stft = librosa.feature.chroma_stft(y=audio_y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)

    def update_path_cost(self, ref_pointer, query_pointer):
        if self.verbose:
            print(
                f"ref_pointer: {ref_pointer}, query_pointer: {query_pointer}, ref shape: {self.ref_stft[:, :ref_pointer].shape} query shape: {self.query_stft[:, :query_pointer].shape}"
            )

        # from librosa
        D, wp = librosa.sequence.dtw(
            X=self.ref_stft[:, :ref_pointer],
            Y=self.query_stft[:, :query_pointer],
            global_constraints=True,
            # subseq=True,
        )
        self.cost_matrix = D
        self.warping_path = wp
        self.warping_path_time = librosa.frames_to_time(
            frames=self.warping_path, hop_length=self.hop_length
        )

    def select_next_direction(self):
        if self.run_count > self.max_run_count:
            if self.previous_direction == Direction.REF:
                next_direction = Direction.QUERY
            else:
                next_direction = Direction.REF

        last_ref_path, last_query_path = self.warping_path[0]
        if (
            last_ref_path + 1 == self.ref_pointer
            and last_query_path + 1 == self.query_pointer
        ):
            next_direction = Direction.BOTH
        elif last_ref_path < last_query_path:
            next_direction = Direction.QUERY
        elif last_ref_path == last_query_path:
            next_direction = Direction.BOTH
        else:
            next_direction = Direction.REF

        return next_direction

    def get_new_input(self):
        qsize = self.sp.chroma_buffer.qsize()
        if qsize <= 1:
            query_chroma_stft = self.sp.chroma_buffer.get()
        else:
            query_chroma_stft = np.hstack([self.sp.chroma_buffer.get() for _ in range(qsize)])
        self.current_query_stft = query_chroma_stft
        self.time_length = self.current_query_stft.shape[1]

        self.query_stft = (
            np.concatenate((self.query_stft, self.current_query_stft), axis=1)
            if self.query_stft.any()
            else self.current_query_stft
        )

    def run(self):
        self.sp.run()  # mic ON
        self.query_pointer += int(CHUNK_SIZE / HOP_LENGTH * self.window_size)
        self.ref_pointer += int(CHUNK_SIZE / HOP_LENGTH * self.window_size)
        start_time = time.time()
        self.get_new_input()
        self.update_path_cost(self.ref_pointer, self.query_pointer)

        while self.sp.is_mic_open:
            if self.select_next_direction() is not Direction.REF:
                self.query_pointer += self.time_length
                self.get_new_input()
                self.update_path_cost(self.ref_pointer, self.query_pointer)

            if self.select_next_direction() is not Direction.QUERY:
                self.ref_pointer += self.time_length
                self.update_path_cost(self.ref_pointer, self.query_pointer)

            if self.select_next_direction() == self.previous_direction:
                self.run_count += 1
            else:
                self.run_count = 1

            if self.select_next_direction() is not Direction.BOTH:
                self.previous_direction = self.select_next_direction()
            self.iteration += 1

        end_time = time.time()
        print(f"duration: {end_time - start_time}")
        self.sp.stop()
    
    def cleanup(self):
        self.query_pointer = 0
        self.ref_pointer = 0
        self.time_length = 0
        self.distance = 0
        self.run_count = 0
        self.previous_direction = None
        self.current_query_stft = None  # (12, N)
        self.query_stft = np.array([])  # (12, n)
        self.query_audio = np.array([])
        self.index1s = np.array([])
        self.index2s = np.array([])
        self.warping_path = None
        self.warping_path_time = None
        self.cost_matrix = None
        self.iteration = 0
