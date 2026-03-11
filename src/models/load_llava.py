from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random


def load_llava(questions: list[str], image_urls: list[str], checkpoint_path: str, model_name: str = "llava-hf/llava-1.5-7b-hf", random_direction: bool = False) -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=14928,
        max_num_seqs=2,
        limit_mm_per_prompt={'image': len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        download_dir=checkpoint_path,
    )


    placeholders = "<image>" * len(image_urls)
    
    prompt = f"USER: {placeholders}\n{questions}\nASSISTANT:"

    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]


    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )
