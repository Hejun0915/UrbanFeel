from vllm.utils import FlexibleArgumentParser
from models.model_list import model_example_map
from tasks.task_import import task_map

def parse_args():
    parser = FlexibleArgumentParser(
        description='Initialize model and task.')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen2_5_vl",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--model-name',
                        type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='Huggingface "model_name".')
    parser.add_argument("--task-name",
                        type=str,
                        default="chat",
                        choices=task_map.keys(),
                        help="The method to run.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing.")
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=2,
        help="Number of images to use for the demo.")
    parser.add_argument("--image-folder",
                        type=str,
                        required=True,
                        help="The local path of the images to use for the demo.")
    parser.add_argument("--output-path",
                        type=str,
                        default='UrbanFeel/output',
                        help="The local path of the output file.")
    parser.add_argument("--gpus-num",
                        type=int,
                        default=1,
                        help="The number of GPUs to use for the demo.")
    
    parser.add_argument("--json-path",
                        type=str,
                        default=None,
                        help="The path of the json file to input the image urls.")
    
    parser.add_argument("--checkpoint-path",
                        type=str,
                        nargs="?",
                        const='./checkpoints',
                        default='./checkpoints',
                        help="The path of the checkpoints files will be save.")
    return parser.parse_args()