import time

from collections import deque
import librosa
from transitions import Machine

from ..config import HUMAN_PLAYER
from ..models import Piece, SubPiece
from .dto import Schedule
from .utils import get_audio_path_from_midi_path, get_midi_from_piece
from .online_dtw import OnlineTimeWarping
from .midiport import midi_port
from .stream_processor import sp
from ..config import HOP_LENGTH, SAMPLE_RATE


class InteractivePerformer:
    states = ["asleep", "following", "playing"]

    def __init__(self, piece: Piece):
        self.piece = piece
        self.schedules = deque(
            Schedule(player=s.player, subpiece=s.subpiece) for s in piece.schedules
        )
        self.current_schedule: Schedule = self.schedules.popleft()
        self.current_player = self.current_schedule.player
        self.current_subpiece: SubPiece = self.current_schedule.subpiece

        self.machine = Machine(
            model=self, states=InteractivePerformer.states, initial="asleep"
        )
        self.machine.add_transition(
            trigger="start_performance",
            source="asleep",
            dest="following",
            conditions="is_human_pianist_playing",
            after="start_following",
        )
        self.machine.add_transition(
            trigger="move_to_next",
            source=["following", "playing"],
            dest="playing",
            unless="is_human_pianist_playing",
            before="cleanup_following",
            after="start_playing",
        )
        self.machine.add_transition(
            trigger="move_to_next",
            source=["playing", "following"],
            dest="following",
            conditions="is_human_pianist_playing",
            after="start_following",
        )
        self.machine.add_transition(
            trigger="start_performance",
            source="asleep",
            dest="playing",
            unless="is_human_pianist_playing",
            after="start_playing",
        )
        self.machine.add_transition(
            trigger="stop_performance",
            source=["following", "playing", "asleep"],
            dest="asleep",
            before="force_quit",
        )
        self.current_timestamp = 0
        self.force_quit_flag = False

    def is_human_pianist_playing(self):
        return self.current_player == HUMAN_PLAYER

    def switch(self):
        if not self.schedules or self.force_quit_flag:
            print("stop performance!")
            self.stop_performance()
            return

        self.current_schedule = self.schedules.popleft()
        self.current_player = self.current_schedule.player
        self.current_subpiece: SubPiece = self.current_schedule.subpiece

        self.move_to_next()  # trigger

    def cleanup_following(self):
        print(f"cleanup following!, current subpiece: {self.current_subpiece}")
        self.odtw.cleanup()

    def start_following(self):
        self.force_quit_flag = False
        print(f"start following!, current subpiece: {self.current_subpiece}")

        current_subpiece_audio_path = get_audio_path_from_midi_path(
            self.current_subpiece.path
        )
        # duration = librosa.get_duration(filename=current_subpiece_audio_path)

        # replace alignment
        self.odtw = OnlineTimeWarping(
            sp,
            ref_audio_path=current_subpiece_audio_path.as_posix(),
            window_size=int(SAMPLE_RATE / HOP_LENGTH) - 1,
            hop_length=HOP_LENGTH,
            verbose=False,
        )
        self.odtw.run()
        # time.sleep(duration)

        print("switch player!")
        self.switch()

    def start_playing(self):
        self.force_quit_flag = False
        print(f"start_playing!, current subpiece: {self.current_subpiece}")
        midi = get_midi_from_piece(self.current_subpiece)
        print(f"play {self.current_subpiece} start")
        midi_port.send(midi)
        print(f"play {self.current_subpiece} end")

        self.switch()

    def force_quit(self):
        self.force_quit_flag = True
        self.cleanup_following()
        self.schedules = deque(
            Schedule(player=s.player, subpiece=s.subpiece) for s in self.piece.schedules
        )
        self.current_schedule: Schedule = self.schedules.popleft()
        self.current_player = self.current_schedule.player
        self.current_subpiece: SubPiece = self.current_schedule.subpiece
        print("force quit & cleanup completed.")
