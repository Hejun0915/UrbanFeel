from models.model_import import *
from utils.image_fetch import fetch_local_image_north, fetch_local_image
from utils.tools import get_random

def load_minicpmv(question: str, image_urls: list[str],checkpoint_path: str, model_name: str="openbmb/MiniCPM-V-2_6", random_direction: bool = False) -> ModelRequestData:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={'image': len(image_urls)},
        download_dir=checkpoint_path
    )

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]


    placeholders = "(<image>./</image>)" * len(image_urls)

    messages = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{placeholders}\n{question}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    if random_direction is False:
        image_data = [fetch_local_image(url) for url in image_urls]

    else:
        image_data = [fetch_local_image_north(url, random_percentage=get_random()) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=messages,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
    )