from models.model_import import *
from utils.image_fetch import fetch_local_image

def load_aria(question: str, image_urls: list[str], checkpoint_path: str, model_name: str = "rhymes-ai/Aria") -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        max_model_len=16384,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": len(image_urls)},
        download_dir=checkpoint_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    placeholders = "<fim_prefix><|img|><fim_suffix>\n" * len(image_urls)
    prompt = (f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_local_image(url) for url in image_urls],
        stop_token_ids= stop_token_ids   
    )