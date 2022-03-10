import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import asyncio
# from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from typing import Optional

from ..config import HOP_LENGTH, CHUNK_SIZE, Direction
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


class OnlineDTW:
    def __init__(
        self,
        sp: StreamProcessor,
        ref_cens,
        window_size,
        max_run_count=30,
        hop_length=HOP_LENGTH,
    ):
        self.sp = sp
        self.ref_cens = ref_cens  # (12, M)
        self.window_size = window_size
        self.max_run_count = max_run_count
        self.hop_length = hop_length
        self.query_pointer = 0
        self.ref_pointer = 0
        self.time_length = 0
        self.distance = 0
        self.run_count = 0
        self.previous_direction = None
        self.current_query_cens = None  # (12, N)
        self.query_cens = None  # (12, n)
        self.query_audio = np.array([])
        self.index1s = np.array([])
        self.index2s = np.array([])
        self.warping_path = None
        self.warping_path_time = None
        self.cost_matrix = None
        self.iteration = 0

    def update_path_cost(self, ref_pointer, query_pointer):
        print(f"ref_pointer: {ref_pointer}, query_pointer: {query_pointer}, ref shape: {self.ref_cens[:, :ref_pointer].shape} query shape: {self.query_cens[:, :query_pointer].shape}")
        # from librosa
        D, wp = librosa.sequence.dtw(
            X=self.ref_cens[:, :ref_pointer],
            Y=self.query_cens[:, :query_pointer],
            global_constraints=True,
        )
        self.cost_matrix = D
        self.warping_path = wp
        self.warping_path_time = librosa.frames_to_time(
            frames=self.warping_path, hop_length=self.hop_length
        )

    def select_next_direction(self):
        if self.run_count > self.max_run_count:
            if self.previous_direction == Direction.REF:
                # time.sleep(0.5)
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
        data_block = b"".join([self.sp.buffer.get() for _ in range(self.window_size)])
        query_audio = np.frombuffer(
            data_block, dtype=np.float32
        )  # length: window_size * 2048
        self.query_audio = np.concatenate((self.query_audio, query_audio))
        query_cens = librosa.feature.chroma_cens(
            y=query_audio,
            hop_length=HOP_LENGTH,
        )  # hop_length: 256
        self.current_query_cens = query_cens
        self.time_length = self.current_query_cens.shape[1]

        if self.query_cens is None:
            self.query_cens = self.current_query_cens
        else:
            self.query_cens = np.concatenate(
                (self.query_cens, self.current_query_cens), axis=1
            )

    def run(self):
        self.sp.run()  # mic ON
        self.query_pointer += int(CHUNK_SIZE / HOP_LENGTH * self.window_size) + 1
        self.ref_pointer += int(CHUNK_SIZE / HOP_LENGTH * self.window_size) + 1
        start_time = time.time()
        self.get_new_input()
        self.update_path_cost(self.ref_pointer, self.query_pointer)

        while self.sp.is_mic_open:
            passed = time.time() - start_time
            print(f"{passed} sec passed")
            if self.warping_path_time[0][1] > passed:
                time.sleep(0.2)
                print("pause for new input")
                continue

            # if passed
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
