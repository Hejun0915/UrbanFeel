from models.model_import import *
from utils.image_fetch import fetch_local_image


def load_idefics3(question: str, image_urls: list[str], checkpoint_path: str, model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3") -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={
            "size": {
                "longest_edge": 2 * 364
            },
        },
        download_dir=checkpoint_path
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|begin_of_text|>User:{placeholders}\n{question}<end_of_utterance>\nAssistant:"  # noqa: E501
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[fetch_local_image(url) for url in image_urls],
    )