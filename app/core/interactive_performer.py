import time

from collections import deque
import librosa
from transitions import Machine

from ..config import AI_PLAYER, HUMAN_PLAYER
from ..models import Piece, SubPiece, Schedule
from .helper import get_audio_path_from_midi_path, get_midi_from_piece
from .online_dtw import OnlineTimeWarping
from .midiport import midi_port


class InteractivePerformer:
    states = ["asleep", "following", "playing"]

    def __init__(self, piece: Piece, oltw: OnlineTimeWarping):
        self.piece = piece
        self.oltw = oltw
        self.schedules = deque(piece.schedules)
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
            trigger="switch",
            source=["following", "playing"],
            dest="playing",
            unless="is_human_pianist_playing",
            before="cleanup_following",
            after="start_playing",
        )
        self.machine.add_transition(
            trigger="switch",
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
            source=["following", "playing"],
            dest="asleep",
        )
        self.current_timestamp = 0
        print("initialize end!")

    def is_human_pianist_playing(self):
        return self.current_player == HUMAN_PLAYER

    def move_to_next(self):
        if not self.schedules:
            print("stop performance!")
            self.stop_performance()
            return

        self.current_schedule = self.schedules.popleft()
        self.current_player = self.current_schedule.player
        self.current_subpiece: SubPiece = self.current_schedule.subpiece

        self.switch()  # trigger

    def cleanup_following(self):
        print(f"cleanup following!, current subpiece: {self.current_subpiece}")
        self.oltw.cleanup()

    def start_following(self):
        print(f"start following!, current subpiece: {self.current_subpiece}")

        current_subpiece_audio_path = get_audio_path_from_midi_path(
            self.current_subpiece.path
        )
        duration = librosa.get_duration(filename=current_subpiece_audio_path)

        # replace alignment
        # self.oltw.run()
        time.sleep(duration)

        print("switch player!")
        self.move_to_next()

    def start_playing(self):
        print(f"start_playing!, current subpiece: {self.current_subpiece}")
        midi = get_midi_from_piece(self.current_subpiece)
        print(f"play {self.current_subpiece} start")
        midi_port.send(midi)
        print(f"play {self.current_subpiece} end")

        self.move_to_next()
