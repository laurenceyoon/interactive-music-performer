from enum import IntEnum


class Direction(IntEnum):
    REF = 1
    QUERY = 2
    BOTH = REF | QUERY


SAMPLE_RATE = 22050
CHUNK_SIZE = 4096
CHANNELS = 1
HOP_LENGTH = 1024
N_FFT = 2048
SOUND_FONT_PATH = "~/Library/Audio/Sounds/Banks/GeneralUser\ GS\ v1.471.sf2"
AI_PLAYER = "VirtuosoNet"
HUMAN_PLAYER = "Pianist"
