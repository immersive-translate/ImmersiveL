from flask import Flask, request, jsonify
from flask import make_response
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 从配置文件加载配置
model_name = app.config["MODEL_NAME"]
gen_params = app.config["GEN_PARAMS"]
mapping = app.config["MAPPING"]
PROMPT_DICT = app.config["PROMPT_DICT"]
os.environ["CUDA_VISIBLE_DEVICES"] = app.config["CUDA_VISIBLE_DEVICES"]
torch.set_num_threads(app.config["NUM_THREADS"])

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16)
model.eval()
model.cuda()  # Ensure the model uses GPU

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
    # 适用于Windows环境下的启动命令
    # ps aux | grep app_win.py
    # 启动命令：python app_win.py

    port = app.config["DEFAULT_PORT"]
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
