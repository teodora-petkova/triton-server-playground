from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationModel:
    name: str
    labels: List[str]
    input_name: str = 'input'
    output_name: str = 'output'
