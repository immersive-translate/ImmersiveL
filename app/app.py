from flask import Flask, request, jsonify
from flask import make_response
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import deepspeed
import requests

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 从配置文件加载配置
model_name = app.config["MODEL_NAME"]
PROMPT_DICT = app.config["PROMPT_DICT"]
gen_params = {
    "return_dict_in_generate": True,
    "output_scores": True,
    "max_new_tokens": 500,
    "do_sample": False,
    "temperature": 0,
    "no_repeat_ngram_size": 20,
    "repetition_penalty": 1.4,
    "length_penalty": 0.9,
}
mapping = app.config["MAPPING"]

os.environ["CUDA_VISIBLE_DEVICES"] = app.config["CUDA_VISIBLE_DEVICES"]
os.environ['TORCH_DISTRIBUTED_DEFAULT_PORT'] = app.config["TORCH_DISTRIBUTED_DEFAULT_PORT"]

torch.set_num_threads(app.config["NUM_THREADS"])


# 读取或创建本地SHA文件
def read_or_create_sha_file():
    if os.path.exists('sha.txt'):
        with open('sha.txt', 'r') as f:
            return f.read().strip()
    else:
        return None

# 写入新的SHA到本地文件


def write_sha_to_file(new_sha):
    with open('sha.txt', 'w') as f:
        f.write(new_sha)

# 获得最新的SHA


def get_latest_sha(model_name):
    response = requests.get(f"https://huggingface.co/api/models/{model_name}")
    if response.status_code == 200:
        remote_sha = response.json().get("sha")
        return remote_sha.strip()
    else:
        print(f"Failed to check for updates: {response.content}")
        return None


# 读取本地SHA
local_sha = read_or_create_sha_file()
# 获取远程SHA
remote_sha = get_latest_sha(model_name)

# 是否需要更新
should_update = local_sha is None or local_sha != remote_sha

# 如果需要更新或者是第一次运行，则下载模型
if should_update:
    print("Downloading model..., this might take a while")
    print(f"- Model name: {model_name}")

orgin_model = AutoModelForCausalLM.from_pretrained(
    model_name, force_download=should_update)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, force_download=should_update)

# 更新本地SHA文件
if remote_sha:
    write_sha_to_file(remote_sha)


model = deepspeed.init_inference(
    model=orgin_model,      # Transformers模型
    mp_size=1,        # GPU数量
    dtype=torch.float16,  # 权重类型(fp16)
    replace_method="auto",  # 让DS自动替换层
    replace_with_kernel_inject=True,  # 使用kernel injector替换
)

# 根据task生成模型输入


def generate_input_prompt(text, task, terms=None):
    terms_prompt = ""
    if terms:
        terms_prompt = "#需应用术语:\n"
        for term in terms:
            terms_prompt += f"{term['src']}\t{term['tag']}\t{term['tgt']}\n"

    return PROMPT_DICT[task].format(input=text, terms=terms_prompt)


@app.route("/translate", methods=["POST"])
@app.route("/v1/translate", methods=["POST"])
def get_translation():
    content = request.json
    text = content['text']
    task = content['task']
    terms = content.get('terms', None)

    # 生成模型输入
    prompt = generate_input_prompt(text, task, terms)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()

    # 执行模型生成
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


def translate_task(source_lang, target_lang):
    return mapping.get((source_lang, target_lang))


@app.route("/immersive_translate", methods=["POST"])
@app.route("/v1/immersive_translate", methods=["POST"])
def immersive_translation():
    content = request.json
    source_lang = content['source_lang']
    target_lang = content['target_lang']
    text_list = content['text_list']

    # 根据语言对选择对应的task
    task = translate_task(source_lang, target_lang)
    # 如果没有对应的task，返回空
    if not task:
        translations = [{'detected_source_lang': None, 'text': None}
                        for _ in text_list]
        ret = {'translations': translations}
        response = make_response(jsonify(ret))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    # 生成模型输入
    prompts = [generate_input_prompt(text, task) for text in text_list]
    inputs = tokenizer(prompts, return_tensors="pt",
                       padding=True, truncation=True)
    input_ids = inputs.input_ids.cuda()

    gen_params["input_ids"] = input_ids
    outputs = model.generate(**gen_params)

    translations = []
    # 从输出中提取翻译结果
    for idx, s in enumerate(outputs.sequences):
        translation = tokenizer.decode(s, skip_special_tokens=True)
        if translation.startswith(prompts[idx]):
            translation = translation[len(prompts[idx]):]
        translations.append({
            'detected_source_lang': source_lang,
            'text': translation
        })

    ret = {
        'translations': translations
    }

    response = make_response(jsonify(ret))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


if __name__ == "__main__":
    # ps aux | grep app.py
    # 启动命令：deepspeed --num_gpus 1 app.py

    port = app.config["DEFAULT_PORT"]
    app.run(host="0.0.0.0", port=port, debug=False,
            use_reloader=False, threaded=False)
