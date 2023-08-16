# ImmersiveL

ImmersiveL is a framework and models hub for connecting languages all over the world open and free.

Now, ImmersiveL app is a Chinese-English bidirectional translation framework based on Deepspeed. The primary structure is found within the app directory, composed of a Python 3.8+ environment with Flask, Deepspeed, and PyTorch.

**The first model for now are trained on a bloomz model, its license can be found at [here](https://bigscience.huggingface.co/blog/the-bigscience-rail-license). The Apache License are licensed to the derived part of the model and other source code file in this repo.**

🌐 **Read in [Chinese (中文)](README_CN.md)**

## Getting Started

1. **Clone the Repository and Set Up the Environment**

   Start by cloning the ImmersiveL repository to your local machine:

   ```bash
   git clone https://github.com/immersive-translate/ImmersiveL.git
   ```

   After cloning, navigate to the `app` directory and install all the necessary packages listed in `requirements.txt`:

   ```bash
   cd ImmersiveL/app
   pip install -r requirements.txt
   ```

2. **Run the Application**

   If you're using a Linux environment, start the application using Deepspeed with the following command:

   ```bash
   deepspeed --num_gpus 1 app.py
   ```

   For Windows users:

   ```bash
   python app_win.py
   ```

   Once you see a message similar to `* Running on [IP Address]`, it indicates that the application has successfully started.

3. **Test the Deployment**

   Now that your application is up and running, test it by translating from Chinese to English:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text": "欧洲经济增长仍面临较大挑战", "task": "zh2en"}' http://localhost:7000/v1/translate
   ```

   If you get a translated result, it indicates that your deployment was successful.

## API Endpoints

### 1. Immersive Translation (`/v1/immersive_translate`)

#### Request

- **Method:** POST
- **Content-Type:** application/json

```json
{
    "source_lang": "zh-CN",
    "target_lang": "en",
    "text_list": [
        "这是一个测试句子",
        "欧洲经济增长仍面临较大挑战"
    ]
}
```

- `source_lang`: Source language code.
- `target_lang`: Target language code.
- `text_list`: An array of text strings to be translated.

#### Response

For the given request:

```json
{
    "translations": [
        {
            "detected_source_lang": "zh-CN",
            "text": "This is a test sentence"
        },
        {
            "detected_source_lang": "zh-CN",
            "text": "Economic growth in Europe continues to face significant challenges"
        }
    ]
}
```

- `translations`: An array containing:
  - `detected_source_lang`: The detected language code of the translated text.
  - `text`: The translated text.

#### Language Codes

- `zh-CN`: Simplified Chinese
- `en`: English

### 2. Basic Translation (`/v1/translate`)

- **Method:** POST
- **Content-Type:** application/json

#### Request Parameters

- `text`: The text you wish to translate.
- `task`: Defines the translation direction. Use "zh2en" for Chinese to English, and "en2zh" for English to Chinese.

#### Response Parameters

- `data`:
  - `translation`: The translated output text.
  - `info`: Additional details about the request, including the model used, original text, etc.
