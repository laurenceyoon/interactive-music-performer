import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional
from functools import partial
import scipy

from ..config import HOP_LENGTH, CHUNK_SIZE, Direction, N_FFT, SAMPLE_RATE
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
MAX_LEN = int(1e4)


class OnlineTimeWarping:
    def __init__(
        self,
        sp: StreamProcessor,
        ref_audio_path,
        window_size,
        max_run_count=30,
        hop_length=HOP_LENGTH,
        verbose=True,
    ):
        self.sp = sp
        self.ref_audio_file = ref_audio_path
        self.window_size = window_size
        self.max_run_count = max_run_count
        self.hop_length = hop_length
        self.frame_per_seg = int(sp.chunk_size / hop_length)
        self.verbose = verbose
        self.ref_pointer = 0
        self.time_length = 0
        self.distance = 0
        self.run_count = 0
        self.previous_direction = None
        self.current_query_stft = None  # (12, n)
        self.query_stft = np.zeros((12, MAX_LEN))  # (12, N) stft of total query
        self.query_audio = np.array([])
        self.index1s = np.array([])
        self.index2s = np.array([])
        self.warping_path = []
        self.cost_matrix = None
        self.dist_matrix = None
        self.acc_dist_matrix = None
        self.acc_len_matrix = None
        self.candidate = None
        self.candi_history = []
        self.iteration = 0
        self.acc_direction = None
        self.cost_matrix_offset = [0, 0]  # (ref, query)
        self.query_pointer = 0
        self.w = self.frame_per_seg * self.window_size

        self.initialize_ref_audio(ref_audio_path)

    def offset(self):
        offset_x = max(self.ref_pointer - self.w, 0)
        offset_y = max(self.query_pointer - self.w, 0)
        return np.array([offset_x, offset_y])

    def local_to_global_index(self, coord):
        return coord + self.offset()

    def global_to_local_index(self, coord):
        return coord - self.offset()

    def initialize_ref_audio(self, audio_path):
        audio_y, sr = librosa.load(audio_path)
        self.ref_audio = audio_y
        ref_stft = librosa.feature.chroma_stft(
            y=audio_y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT, norm=1
        )
        ref_len = ref_stft.shape[1]
        truncated_len = (
            (ref_len - 1) // self.frame_per_seg
        ) * self.frame_per_seg  # initialize_ref_audio 에서 ref_stft 길이가 frame_per_seg (4) 로 나눠지게 마지막을 버림
        self.ref_stft = ref_stft[:, :truncated_len]

        self.global_cost_matrix = np.zeros(
            (self.ref_stft.shape[1] * 2, self.ref_stft.shape[1] * 2)
        )

    def init_dist_matrix(self):
        print("init")
        ref_stft_seg = self.ref_stft[:, : self.ref_pointer]  # [F, M]
        query_stft_seg = self.query_stft[:, : self.query_pointer]  # [F, N]
        dist = scipy.spatial.distance.cdist(ref_stft_seg.T, query_stft_seg.T)

        if self.verbose:
            print(
                f"ref_stft_seg: {ref_stft_seg.shape}, query_stft_seg: {query_stft_seg.shape}, dist: {dist.shape}, dist_matrix shape: {self.dist_matrix.shape}"
            )
        w = self.window_size * self.frame_per_seg
        self.dist_matrix[w - dist.shape[0] :, w - dist.shape[1] :] = dist

    def update_dist_matrix(self, direction: Direction):
        if self.verbose:
            print(f"update_path_cost with direction: {direction.name}")
        x = self.ref_pointer
        y = self.query_pointer
        w = self.window_size * self.frame_per_seg

        ref_stft_seg = self.ref_stft[:, max(x - w, 0) : x]  # [F, M]
        query_stft_seg = self.query_stft[:, max(y - w, 0) : y]  # [F, N]
        dist = scipy.spatial.distance.cdist(ref_stft_seg.T, query_stft_seg.T)
        if self.verbose:
            print(
                f"current pointer: {(x, y)}, ref shape: {ref_stft_seg.shape} query shape: {query_stft_seg.shape}, dist shape: {dist.shape}, dist_matrix shape: {self.dist_matrix.shape}"
            )

        self.dist_matrix[
            w - dist.shape[0] :, w - dist.shape[1] :
        ] = dist  # dist_matrix 끝 점에 맞춰서 항상 업데이트 되도록 함.

    def init_matrix(self):
        x = self.ref_pointer
        y = self.query_pointer
        w = self.w
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))
        x_seg = self.ref_stft[:, x - wx : x].T  # [wx, 12]
        y_seg = self.query_stft[:, y - d : y].T  # [d, 12]
        dist = scipy.spatial.distance.cdist(x_seg, y_seg)  # [wx, d``]

        for i in range(wx):
            for j in range(d):
                local_dist = dist[i, j]
                update_x0 = 0
                update_y0 = wy - d
                if i == 0 and j == 0:
                    new_acc[i, j] = local_dist
                elif i == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i, update_y0 - 1]
                    new_len_acc[i, update_y0 + j] = 1 + new_len_acc[i, update_y0 - 1]
                elif j == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i - 1, update_y0]
                    new_len_acc[i, update_y0 + j] = (
                        local_dist + new_len_acc[i - 1, update_y0]
                    )
                else:
                    compares = [
                        new_acc[i - 1, update_y0 + j],
                        new_acc[i, update_y0 + j - 1],
                        new_acc[i - 1, update_y0 + j - 1],
                    ]
                    len_compares = [
                        new_len_acc[i - 1, update_y0 + j],
                        new_len_acc[i, update_y0 + j - 1],
                        new_len_acc[i - 1, update_y0 + j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[i, update_y0 + j] = local_dist + compares[local_direction]
                    new_len_acc[i, update_y0 + j] = 1 + len_compares[local_direction]
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc
        self.select_candidate()

    def update_accumulate_matrix(self, direction):
        # local cost matrix
        x = self.ref_pointer
        y = self.query_pointer
        w = self.w
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))

        if direction == Direction.REF:
            new_acc[:-d, :] = self.acc_dist_matrix[d:]
            new_len_acc[:-d, :] = self.acc_len_matrix[d:]
            x_seg = self.ref_stft[:, x - d : x].T  # [d, 12]
            y_seg = self.query_stft[:, y - wy : y].T  # [wy, 12]
            dist = scipy.spatial.distance.cdist(x_seg, y_seg)  # [d, wy]

            for i in range(d):
                for j in range(wy):
                    local_dist = dist[i, j]
                    update_x0 = wx - d
                    update_y0 = 0
                    if j == 0:
                        new_acc[update_x0 + i, j] = (
                            local_dist + new_acc[update_x0 + i - 1, j]
                        )
                        new_len_acc[update_x0 + i, j] = (
                            new_len_acc[update_x0 + i - 1, j] + 1
                        )
                    else:
                        compares = [
                            new_acc[update_x0 + i - 1, j],
                            new_acc[update_x0 + i, j - 1],
                            new_acc[update_x0 + i - 1, j - 1],
                        ]
                        len_compares = [
                            new_len_acc[update_x0 + i - 1, j],
                            new_len_acc[update_x0 + i, j - 1],
                            new_len_acc[update_x0 + i - 1, j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[update_x0 + i, j] = (
                            local_dist + compares[local_direction]
                        )
                        new_len_acc[update_x0 + i, j] = (
                            1 + len_compares[local_direction]
                        )

        elif direction == Direction.QUERY:
            overlap_y = wy - d
            new_acc[:, :-d] = self.acc_dist_matrix[:, -overlap_y:]
            new_len_acc[:, :-d] = self.acc_len_matrix[:, -overlap_y:]
            x_seg = self.ref_stft[:, x - wx : x].T  # [wx, 12]
            y_seg = self.query_stft[:, y - d : y].T  # [d, 12]
            dist = scipy.spatial.distance.cdist(x_seg, y_seg)  # [wx, d``]

            for i in range(wx):
                for j in range(d):
                    local_dist = dist[i, j]
                    update_x0 = 0
                    update_y0 = wy - d
                    if i == 0:
                        new_acc[i, update_y0 + j] = (
                            local_dist + new_acc[i, update_y0 - 1]
                        )
                        new_len_acc[i, update_y0 + j] = (
                            1 + new_len_acc[i, update_y0 - 1]
                        )
                    else:
                        compares = [
                            new_acc[i - 1, update_y0 + j],
                            new_acc[i, update_y0 + j - 1],
                            new_acc[i - 1, update_y0 + j - 1],
                        ]
                        len_compares = [
                            new_len_acc[i - 1, update_y0 + j],
                            new_len_acc[i, update_y0 + j - 1],
                            new_len_acc[i - 1, update_y0 + j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[i, update_y0 + j] = (
                            local_dist + compares[local_direction]
                        )
                        new_len_acc[i, update_y0 + j] = (
                            1 + len_compares[local_direction]
                        )
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc

    def calculate_warping_path(self, start, end):
        """calculate each warping path from start to end point and return distance"""
        wp = None
        distance = None
        return wp, distance

    def update_warping_path(self):
        table = self.cost_matrix
        i = self.cost_matrix.shape[0] - 1
        j = (
            self.cost_matrix.shape[1] - 1
        )  # start = (i, j), end = (ref_until, query_until)

        ref_until = 0
        query_until = 0

        offset = self.offset()
        if offset[0] < 0 or offset[1] < 0:
            ref_until = max(i - self.ref_pointer, 0)
            query_until = max(j - self.query_pointer, 0)

        path = [(i, j)]
        while i > ref_until or j > query_until:
            minval = np.inf
            if table[i - 1, j] < minval:
                minval = table[i - 1, j]
                step = (i - 1, j)
            if table[i][j - 1] < minval:
                minval = table[i, j - 1]
                step = (i, j - 1)
            if table[i - 1][j - 1] < minval:
                minval = table[i - 1, j - 1]
                step = (i - 1, j - 1)
            path.insert(0, step)
            i, j = step
        path += offset
        self.warping_path.extend(path)

    def update_path_cost(self, direction):
        self.update_accumulate_matrix(direction)
        self.select_candidate()
        # self.update_warping_path()

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / self.acc_len_matrix[-1, :]
        norm_y_edge = self.acc_dist_matrix[:, -1] / self.acc_len_matrix[:, -1]
        cat = np.concatenate((norm_x_edge, norm_y_edge))
        min_idx = np.argmin(cat)
        offset = self.offset()
        if min_idx <= len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0], min_idx])
        else:
            self.candidate = np.array(
                [min_idx - len(norm_x_edge), self.query_pointer - offset[1]]
            )

    def save_history(self):
        offset = self.offset()
        self.candi_history.append(offset + self.candidate)

    def select_next_direction(self):
        if self.run_count > self.max_run_count:
            if self.previous_direction == Direction.REF:
                next_direction = Direction.QUERY
            else:
                next_direction = Direction.REF

        offset = self.offset()
        x0 = offset[0]
        y0 = offset[1]
        if self.candidate[0] == self.ref_pointer - x0:
            # ref direction
            next_direction = Direction.REF
        else:
            assert self.candidate[1] == self.query_pointer - y0
            next_direction = Direction.QUERY
        self.save_history()
        return next_direction

        # return Direction.BOTH

    def get_new_input(self):
        #  get only one input
        query_chroma_stft = self.sp.chroma_buffer.get()["chroma_stft"]
        self.current_query_stft = query_chroma_stft
        q_length = self.current_query_stft.shape[1]
        self.query_stft[
            :, self.query_pointer : self.query_pointer + q_length
        ] = query_chroma_stft
        self.query_pointer += q_length
        if self.verbose:
            print(f"updated q_index: {self.query_pointer}, q_length: {q_length}")

    def _check_run_time(self, start_time, duration):
        return time.time() - start_time < duration

    def run(self, fig=None, h=None, hfig=None, duration=None, fake=False):
        self.sp.run(fake)  # mic ON
        start_time = time.time()

        self.ref_pointer += self.window_size * self.frame_per_seg
        self.get_new_input()
        self.init_matrix()

        run_condition = (
            partial(self._check_run_time, start_time, duration)
            if duration is not None
            else self.sp.is_open
        )
        while run_condition() and self.ref_pointer <= (
            self.ref_stft.shape[1] - self.frame_per_seg
        ):
            if self.verbose:
                print(f"\niteration: {self.iteration}")
            direction = self.select_next_direction()

            if direction is Direction.QUERY:
                self.get_new_input()
                self.update_path_cost(direction)

            elif direction is Direction.REF:
                self.ref_pointer += self.frame_per_seg
                self.update_path_cost(direction)

            if direction == self.previous_direction:
                self.run_count += 1
            else:
                self.run_count = 1

            # if direction is not Direction.BOTH:
            self.previous_direction = direction
            self.iteration += 1

            if duration is None:
                duration = int(librosa.get_duration(filename=self.ref_audio_file)) + 1
            if fig and h and hfig:
                h.set_data(self.query_stft[:, : int(SAMPLE_RATE / HOP_LENGTH) * duration])
                #             clear_output(wait=True)
                hfig.update(fig)

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
