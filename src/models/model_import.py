import os
import re
import random
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

import json
from huggingface_hub import snapshot_download
import PIL
from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
from tqdm import tqdm

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None

