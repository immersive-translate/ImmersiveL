from flask import Flask, request, jsonify
from flask import make_response
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import deepspeed

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 从配置文件加载配置
model_name = app.config["MODEL_NAME"]
os.environ["CUDA_VISIBLE_DEVICES"] = app.config["CUDA_VISIBLE_DEVICES"]
os.environ['TORCH_DISTRIBUTED_DEFAULT_PORT'] = app.config["TORCH_DISTRIBUTED_DEFAULT_PORT"]

torch.set_num_threads(app.config["NUM_THREADS"])

dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

ds_engine = deepspeed.initialize(model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16), config_params=app.config["DS_CONFIG"])[0]
ds_engine.module.eval()
model = ds_engine.module

# 根据task生成模型输入


def generate_input_prompt(text, task, terms=None):
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

    terms_prompt = ""
    if terms:
        terms_prompt = "#需应用术语:\n"
        for term in terms:
            terms_prompt += f"{term['src']}\t{term['tag']}\t{term['tgt']}\n"

    return PROMPT_DICT[task].format(input=text, terms=terms_prompt)


@app.route("/translate", methods=["POST"])
def get_translation():
    content = request.json
    text = content['text']
    task = content['task']
    terms = content.get('terms', None)

    # # 生成模型输入
    prompt = generate_input_prompt(text, task, terms)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()

    # # 执行模型生成
    gen_params = app.config["GEN_PARAMS"]
    gen_params["input_ids"] = input_ids
    outputs = model.generate(**gen_params)
    for s in outputs.sequences:
        translation = tokenizer.decode(s, skip_special_tokens=True)

    # translation替换开头的prompt为空
    if translation.startswith(prompt):
        translation = translation[len(prompt):]

    
    # translation = prompt
    ret = {
        'data': {
            "translation": translation,

            'info': {
                'text': text,
                'task': task,
                'terms': terms,
                'model_name': model_name,
            }
        }
    }

    response = make_response(jsonify(ret))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


if __name__ == "__main__":
    # ps aux | grep app.py
    # 启动命令：deepspeed --num_gpus 1 app.py

    port = app.config["DEFAULT_PORT"]
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)