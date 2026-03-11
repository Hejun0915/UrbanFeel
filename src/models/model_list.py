from models.load_aria import load_aria
from models.load_aya_vision import load_aya_vision
from models.load_deepseek_vl2 import load_deepseek_vl2
from models.load_gemma3 import load_gemma3
from models.load_idefics3 import load_idefics3
from models.load_internvl import load_internvl
from models.load_llava_next import load_llava_next
from models.load_llava import load_llava
from models.load_minicpm import load_minicpmv
from models.load_mistral import load_mistral3vl
from models.load_phi3v import load_phi3v
from models.load_phi4mm import load_phi4mm
from models.load_qwen import load_qwen2_5_vl

model_example_map = {
    "aria": load_aria,
    "aya_vision": load_aya_vision,
    "deepseek_vl_v2": load_deepseek_vl2,
    "gemma3": load_gemma3,
    "idefics3": load_idefics3,
    "internvl_chat": load_internvl,
    "llava_next": load_llava_next,
    "llava_vision": load_llava,
    "minicpm_v": load_minicpmv,
    "mistral3vl": load_mistral3vl,
    "phi3_v": load_phi3v,
    "phi4_mm": load_phi4mm,
    "qwen2_5_vl": load_qwen2_5_vl,
}