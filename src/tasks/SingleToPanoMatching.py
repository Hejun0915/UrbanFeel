from tasks.task_import import os, json, tqdm, args
from tasks.task_import import NamedTuple, Optional
from tasks.task_import import asdict
from tasks.task_import import SamplingParams
from tasks.task_import import LLM
from utils.image_fetch import is_image_corrupted

from models.model_list import model_example_map

QUESTION = "Question: I will show you two street view images. Can you tell me whether the two images are taken from the same place? We consider the area within 50m to be the same place. Your answer should be your choice from the following options: \n\n\n A) Yes  \nB) No." 

def SingleToPanoMatching(model, model_name: str, seed: Optional[int], gpus_num: int, json_path: str, image_folder: str, output_path: str, checkpoint_path: str):
    print(image_folder)
    
    output_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except:
        print("JSON unvalid. Please check your json")
        return
    
    results = []
    
    for idx in tqdm(range(len(json_data))):
        data = json_data[idx]
        image_urls = [
            os.path.join(image_folder, data['pano']),
            os.path.join(image_folder, data['single'])
        ]


        image_status = True
        
        for image_url in image_urls:
            if is_image_corrupted(image_url):
                image_status = False
                break
            
        if image_status == False:
            continue



        if idx == 0:
            req_data = model_example_map[model](question = QUESTION, image_urls = image_urls, checkpoint_path = checkpoint_path,model_name = model_name)
            engine_args = asdict(req_data.engine_args) | {"seed": args.seed, "tensor_parallel_size": gpus_num, "trust_remote_code": True}
            llm = LLM(**engine_args)
            sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=1024,
                                            stop_token_ids=[151643, 151644])

        else:
            req_data = model_example_map[model](question = QUESTION, image_urls = image_urls, checkpoint_path = checkpoint_path,model_name = model_name)
            sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=1024,
                                            stop_token_ids=[151643, 151644])

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
            result_entry = {
                "pano": data['pano'],
                "single": data['single'],
                "gt": data['gt'],
                "response": generated_text.strip()
            }
            results.append(result_entry)


    output_json_path = os.path.join(
        output_path, f'{model_name.split("/")[-1]}_SingleToPanoMatching.json'
    )
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Output has been saved in:{output_json_path}")