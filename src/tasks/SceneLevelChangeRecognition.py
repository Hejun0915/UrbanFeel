from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM

from models.model_list import model_example_map

QUESTION = "Question: You will be shown a pair of street-view images taken at the same location but at different times. These images reflect changes in the urban environment over time. The first image shows the scene in the past, and the second image shows the current scene. Carefully observe the differences between the two images. Then, choose the one most appropriate category that describes the change.\n\nPlease only reply with the letter (A–E) that corresponds to your answer.\n\nA) Road Change (e.g., pavement, markings, or road structure have changed)\nB) Building Facade Change (e.g., wall colors, textures, or decorations have changed)\nC) Building Presence Change (e.g., buildings entity have appeared or disappeared)\nD) Vegetation Change (e.g., trees, grass, or other plants have increased or decreased)\nE) Mobility-Related Change (e.g., differences caused by the movement of vehicles or pedestrians)"

def SceneLevelChangeRecognition(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        json_data = None
        return

    first_round = True

    output_path = os.path.join(output_path, "SceneLevelChangeRecognition")
    os.makedirs(output_path, exist_ok=True)

    results = []

    for data in tqdm(json_data):
        image_urls = [
            os.path.join(image_folder, data["before"]),
            os.path.join(image_folder, data["after"])
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
            output_path, f'{model_name.split("/")[-1]}_SceneLevelChangeRecognition.json'
        )
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Output has been saved in:{output_json_path}")
