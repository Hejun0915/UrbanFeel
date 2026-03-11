from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random

def load_phi3v(question: str, image_urls: list[str], checkpoint_path: str, model_name: str = "microsoft/Phi-3.5-vision-instruct", random_direction: bool = False) -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"num_crops": 4},
        download_dir=checkpoint_path,
    )
    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{question}<|end|>\n<|assistant|>\n"

    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )
