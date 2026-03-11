from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM

from models.model_list import model_example_map

QUESTION = "Question: You are given five street-view images of the same general location. Image 1 was captured more than 10 years ago and serves as a reference image representing the past. Images 2 to 5 were captured more than 10 years later and represent possible future states of the same location after urban transformation.\n\nBased on the visual evidence and possible urban changes, please choose which of Images 2–5 most likely represents the same location as Image 1 after city-level development or renewal.\n\nPlease reply with only the letter A, B, C, or D:\n\nA) Image 2\nB) Image 3\nC) Image 4\nD) Image 5"

def FutureSceneIdentification(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        json_data = None
        return

    first_round = True

    output_path = os.path.join(output_path, "FutureSceneIdentification")
    os.makedirs(output_path, exist_ok=True)

    output_json_path = os.path.join(
        output_path, f'{model_name.split("/")[-1]}_FutureSceneIdentification.json'
    )

    processed_keys = set()
    results = []
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_keys = {item["before"] for item in results if "before" in item}

    for data in tqdm(json_data):
        if data["before"] in processed_keys:
            continue

        image_urls = [
            os.path.join(image_folder, data["before"]),
            os.path.join(image_folder, data["after A"]),
            os.path.join(image_folder, data["after B"]),
            os.path.join(image_folder, data["after C"]),
            os.path.join(image_folder, data["after D"])
        ]

        if first_round:
            req_data = model_example_map[model](question = QUESTION, image_urls = image_urls, checkpoint_path = checkpoint_path,model_name = model_name)
            engine_args = asdict(req_data.engine_args) | {
                "seed": seed,
                "tensor_parallel_size": gpus_num,
                "trust_remote_code": True
            }
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
            processed_keys.add(data["before"])

        first_round = False


        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Output has been saved in:{output_json_path}")
