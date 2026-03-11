from models.model_import import *
from utils.image_fetch import fetch_local_image,fetch_local_image_north
from utils.tools import get_random


def load_internvl(question: str, image_urls: list[str], checkpoint_path: str, model_name: str = "OpenGVLab/InternVL3-2B", random_direction: bool = False) -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        download_dir=checkpoint_path
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
    )
