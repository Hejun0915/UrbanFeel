from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random


def load_gemma3(question: str, image_urls: list[str], checkpoint_path: str, model_name: str = "google/gemma-3-4b-it", random_direction: bool = False) -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_name,  
        max_model_len=16384,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        download_dir=checkpoint_path
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )
