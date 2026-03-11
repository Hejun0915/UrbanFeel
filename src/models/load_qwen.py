from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random

def load_qwen2_5_vl(question: str, image_urls: list[str], checkpoint_path: str, model_name: str, random_direction: bool = False) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384 if process_vision_info is None else 16384,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
        download_dir=checkpoint_path
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]

    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )