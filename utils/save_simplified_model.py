# Save huggingface checkpoint as simplified models
import os
import sys
import torch
sys.path.append(os.getcwd())
from models.mantis import Mantis

model_path = "your_model_path"
mantis = Mantis.from_pretrained(
    model_path,
)
# mantis.load_state_dict(torch.load("model.pt"))
mantis.save_pretrained("save_dir")