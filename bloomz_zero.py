import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7,8,9"

from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import deepspeed
import torch.distributed as dist
from argparse import ArgumentParser
import time
import torch
torch.set_num_threads(12)

#  deepspeed --num_gpus 8  bloomz_zero.py --name "/home/envd/data/timvan/test_extend/pennyfunai-translate/pennyfun/models/bilingual_trans_bloomz_1b1/checkpoint-18000" --batch_size 2

# 建立了一个参数解析器
parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="模型名称")
parser.add_argument("--local_rank", required=False, type=int, help="用于分布式启动器")
parser.add_argument("--batch_size", default=1, type=int, help="批处理大小")
parser.add_argument("--benchmark", action="store_true", help="额外运行基准测试")
parser.add_argument("--cpu_offload", action="store_true", help="是否激活CPU卸载")
parser.add_argument("--nvme_offload_path", help="是否激活NVME卸载及在nvme上的路径")

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")

# local_model_path = "/home/envd/data/timvan/test_extend/pennyfunai-translate/pennyfun/models/bilingual_trans_bloomz_1b1/checkpoint-7300"

dtype = torch.bfloat16

# 模型配置
# 模型加载和在GPU上实例化（通过ZeRO）
model_name = args.name
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

model_hidden_size = config.hidden_size
train_batch_size = 1 * world_size

ds_config = {
    "fp16": {
        "enabled": dtype == torch.float16,
    },
    "bf16": {
        "enabled": dtype == torch.bfloat16,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 0,
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

dschf = HfDeepSpeedConfig(ds_config)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = model.eval()

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

# 开始生成
num_tokens = 100
input_sentences = [
    """下面是一段英文文本，请将它翻译成中文。\n#英文文本:\nDeepSpeed is a machine learning framework\n\n#中文翻译:\n""",
    """下面是一段中文文本，请将它翻译成英文。\n#中文文本:\n联合国教科文组织前副总干事：解决非洲问题，要靠非洲方案，再加中国经验\n\n#英文翻译:\n"""
]
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

inputs = input_sentences[: 1]  # 按需求调整batch size

# 定义生成函数
def generate():
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs)

# 执行生成
pairs = generate()
for i, o in pairs:
    print(f"{'-'*60}\nin={i}\nout={o}\n")
