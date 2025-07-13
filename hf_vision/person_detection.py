import os
import torch
import transformers

from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
# from huggingface_hub import snapshot_download
from PIL import Image

MODEL_ID = "google/paligemma-3b-pt-224"
DATA_DIR = "/data/potw"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(
        f"Using transformers version: {transformers.__version__}\nTorch version: {torch.__version__}"
    )

    # From: https://huggingface.co/google/paligemma-3b-pt-224
    model_id = "google/paligemma-3b-pt-224"
    device = "cuda:0"
    dtype = torch.bfloat16

    #snapshot_download(repo_id=model_id, repo_type="model")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device,
        quantization_config=bnb_config,
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image_list = [
        x for x in os.listdir(DATA_DIR) if x.endswith((".png", ".jpg", ".jpeg"))
    ]
    prompt = "detect person"
    image = Image.open(os.path.join(DATA_DIR, image_list[1])).convert("RGB")

    print(f"Using image: {image_list[1]}")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate output tokens
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,  # Max tokens to generate in the response
            do_sample=False,  # For deterministic output, set to False
            num_beams=1,  # For deterministic output, set to 1
        )

    result_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Raw Model Output:\n{'\n'.join(result_text.split(';'))}\n")
