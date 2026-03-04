from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM

from models.model_list import model_example_map

QUESTION = "Question: You are given four street-view images (Image A, B, C, and D), each taken at a different point in time from the same location. These images reflect different stages of urban development. Your task is to determine the correct chronological order of the images from the **least** to the **most** developed.\n\nTo complete this task, first analyze each image based on visual cues such as building construction, road quality, greenery, public infrastructure, and signs of modernization.\n\nPlease follow this format in your response:\n\n1. **Answer**: List the image order from least to most developed, using this format:\n   [Image X → Image Y → Image Z → Image W]\n\n2. **Reasoning**: Briefly explain why you chose this order, referring to the key urban development features you observed in the images.\n\nLet’s think step by step."

def TemporalSequenceReasoning(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        json_data = None
        return

    first_round = True

    output_path = os.path.join(output_path, "TemporalSequenceReasoning")
    os.makedirs(output_path, exist_ok=True)

    results = []

    for data in tqdm(json_data):
        image_urls = [
            os.path.join(image_folder, data["Time A"]),
            os.path.join(image_folder, data["Time B"]),
            os.path.join(image_folder, data["Time C"]),
            os.path.join(image_folder, data["Time D"])
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
            output_path, f'{model_name.split("/")[-1]}_TemporalSequenceReasoning.json'
        )
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Output has been saved in:{output_json_path}")