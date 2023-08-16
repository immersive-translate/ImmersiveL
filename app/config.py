# config.py
MODEL_NAME = "funstoryai/immersiveL-exp"
CUDA_VISIBLE_DEVICES = "0"
NUM_THREADS = 1
DEFAULT_PORT = 7000
TORCH_DISTRIBUTED_DEFAULT_PORT = "29512"

DS_CONFIG = {
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 256,
        "stage3_prefetch_bucket_size": 256,
        "stage3_param_persistence_threshold": 0,
    },
    "steps_per_print": 2000,
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

PROMPT_DICT = {
    "en2zh": (
        "下面是一段英文文本，请将它翻译成中文。\n"
        "{terms}"
        "#英文文本:\n{input}\n\n#中文翻译:\n"
    ),
    "zh2en": (
        "下面是一段中文文本，请将它翻译成英文。\n"
        "{terms}"
        "#中文文本:\n{input}\n\n#英文翻译:\n"
    ),
}

GEN_PARAMS = {
    "return_dict_in_generate": True,
    "output_scores": True,
    "max_new_tokens": 400,
    "do_sample": False,
    "temperature": 0,
    "num_beams": 48,
    "num_beam_groups": 12,
    "no_repeat_ngram_size": 20,
    "diversity_penalty": 0.1,
    "repetition_penalty": 1.4,
    "length_penalty": 0.9,
}

MAPPING = {
    ("zh-CN", "en"): "zh2en",
    ("en", "zh-CN"): "en2zh"
    # 可以增加其他的语言对
}
