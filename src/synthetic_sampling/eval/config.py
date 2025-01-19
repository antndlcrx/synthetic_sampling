from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch

@dataclass
class EvaluationConfig:
    question_ids: List[str]
    special_codes: Dict[str, Optional[str]]  # map missing values codes from survey to a specified value or None
    profile_prompt_template: str
    model: Any  
    tokenizer: Any 
    device: torch.device
    batch_size: int = 4  
    f1_average: str = "weighted"
