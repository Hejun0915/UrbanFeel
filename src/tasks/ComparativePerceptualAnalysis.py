from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM

from models.model_list import model_example_map


def ComparativePerceptualAnalysis(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        json_data = None
        return

    first_round = True

    output_path = os.path.join(output_path, "ComparativePerceptualAnalysis")
    os.makedirs(output_path, exist_ok=True)

    PerceptionDimensions = ['beautiful', 'wealthy', 'lively', 'safe']


    for dimension in PerceptionDimensions:

        results = []
        QUESTION = "Task: You are given two street-view images of the same location taken at different times. The first image shows the past appearance, and the second image shows the current appearance. Your goal is to assess whether the place has become more {}, less {}, or stayed the same in terms of visual perception.\n\nPlease first select one of the following options:\nA) More {}\nB) Less {}\nC) No Change\n\nThen, briefly explain your reasoning by listing 2 to 3 visual factors that influenced your judgment. \n\nPlease follow this format:\n\nAnswer: <A / B / C>\n\nReasoning:\n1. <first visual factor>\n2. <second visual factor>\n3. <optional third visual factor>".format(dimension, dimension, dimension, dimension)

        dimension_data = [data for data in json_data if data.get("dimension") == dimension]

        for data in tqdm(dimension_data):
            image_urls = [
                os.path.join(image_folder, "PanoData", data["before"]),
                os.path.join(image_folder, "PanoData", data["after"])
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
                output_path, f'{model_name.split("/")[-1]}_ComparativePerceptualAnalysis_{dimension}.json'
            )
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✅ Output has been saved in:{output_json_path}")