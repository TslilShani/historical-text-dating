from enum import StrEnum


class DatasetSplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "eval"
    TEST = "test"
