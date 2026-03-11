from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random

def load_llava_next(questions: list[str], image_urls: list[str], checkpoint_path: str, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf") -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_seqs=2,
        limit_mm_per_prompt={'image': len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        download_dir=checkpoint_path
    )


    placeholders = "<image>" * len(image_urls)
    prompt = f"[INST] {placeholders}\n{questions}[/INST]"
  

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_local_image(url) for url in image_urls],
    )