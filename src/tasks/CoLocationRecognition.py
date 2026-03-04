from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM

from models.model_list import model_example_map

QUESTION = "Question: You are given two single-view street images taken from an urban scene. Based on the layout of the surroundings, building facades, road structure, and overall urban context, determine whether the two images were taken on the same street.\n\nPlease follow this format in your response:\n\n1. **Answer**: Reply with only the letter A or B"

def CoLocationRecognition(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)

    # 读取 json 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        json_data = None
        return

    first_round = True

    output_path = os.path.join(output_path, "CoLocationRecognition")
    os.makedirs(output_path, exist_ok=True)

    results = []

    for data in tqdm(json_data):
        image_urls = [
            os.path.join(image_folder, data["image_type"], data["image 1"]),
            os.path.join(image_folder, data["image_type"], data["image 2"]),
        ]

        if first_round:
            req_data = model_example_map[model](question = QUESTION, image_urls = image_urls, checkpoint_path = checkpoint_path,model_name = model_name)
            engine_args = asdict(req_data.engine_args) | {"seed": seed, "tensor_parallel_size": gpus_num, "trust_remote_code": True}
            llm = LLM(**engine_args)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                stop_token_ids=[151643, 151644]
            )
        else:
            req_data = model_example_map[model](question = QUESTION, image_urls = image_urls, checkpoint_path = checkpoint_path,model_name = model_name)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                stop_token_ids=[151643, 151644]
            )

        outputs = llm.generate(
            {
                "prompt": req_data.prompt,
                "multi_modal_data": {
                    "image": req_data.image_data
                },
            },
            sampling_params=sampling_params,
            lora_request=req_data.lora_requests,
        )

        for o in outputs:
            generated_text = o.outputs[0].text
            data["response"] = generated_text.strip()
            results.append(data)

        first_round = False

        output_json_path = os.path.join(
            output_path, f'{model_name.split("/")[-1]}_CoLocationRecognition.json'
        )
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Output has been saved in:{output_json_path}")