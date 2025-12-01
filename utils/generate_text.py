import os
import sys
sys.path.append(os.getcwd())
from models.mantis import Mantis
from qwen_vl_utils import process_vision_info


model_path = "your_model_path"
mantis = Mantis.from_pretrained(
    model_path,
)

model = mantis.model.mllm_backbone
model = model.to("cuda")
processor = mantis.model.tokenizer

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": "test.png"
            },
            {
                "type": "text", 
                "text": "Describe this picture."
            },
        ],
    },
]

prompts = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=prompts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)