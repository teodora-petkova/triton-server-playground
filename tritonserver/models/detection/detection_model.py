# This script is based on different grpc examples for triton server
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class DetectionModel:
    name: str = 'infamous_symbols'
    input_name: str = 'images'
    output_name: str = 'output0'
    input_hw: Tuple[int, int] = (640, 640)
    conf_thresh: float = 0.35
    iou_thresh: float = 0.45
    fp_mode: str = 'FP32'

    @classmethod
    def get_fp(cls):
        return np.float16 if '16' in cls.fp_mode else np.float32