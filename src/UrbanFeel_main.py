import os
import re
import random
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional
from utils.parse_args import parse_args

from models.model_list import model_example_map
from tasks import task_map 

def main(args:Namespace):
    
    task_name = args.task_name
    model = args.model_type
    model_name = args.model_name
    gpus_num = args.gpus_num
    seed = args.seed
    json_path = args.json_path
    image_folder = args.image_folder
    output_path = args.output_path
    checkpoint_path = args.checkpoint_path
    

    task_func = task_map.get(task_name)
    if task_func:
        task_func(model, model_name, seed=seed, gpus_num=gpus_num, image_folder=image_folder, json_path=json_path, output_path=output_path,checkpoint_path=checkpoint_path)
    else:
        print(f"No task found for : {task_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)