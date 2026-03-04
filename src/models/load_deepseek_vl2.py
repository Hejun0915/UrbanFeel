from models.model_import import *
from utils.image_fetch import fetch_local_image

def load_deepseek_vl2(question: str,
                      image_urls: list[str], checkpoint_path: str, model_name: str = "deepseek-ai/deepseek-vl2-tiny") -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=5,
        dtype="bfloat16",
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": len(image_urls)},
        download_dir=checkpoint_path
    )

    placeholder = "".join(f"image_{i}:<image>\n"
                          for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_local_image(url) for url in image_urls],
    )