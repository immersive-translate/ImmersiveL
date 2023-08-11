# ImmersiveL

ImmersiveL is a framework and models hub for connecting languages all over the world open and free.

Now, ImmersiveL app is a Chinese-English bidirectional translation framework based on Deepspeed. The primary structure is found within the app directory, composed of a Python 3.8+ environment with Flask, Deepspeed, and PyTorch.

**The first model for now are trained on a bloomz model, its license could be found at [here](https://bigscience.huggingface.co/blog/the-bigscience-rail-license). The Apache License are licensed to the derived part of the model and other source code file in this repo.**

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

   Start the application using Deepspeed with the following command:

   ```bash
   deepspeed --num_gpus 1 app.py
   ```

    Once you see a message similar to `* Running on [IP Address]`, it indicates that the application has successfully started.

## Using ImmersiveL

Once the application is up and running, you can easily use the provided translation endpoints.

### Example 1: Translating from Chinese to English

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "欧洲经济增长仍面临较大挑战", "task": "zh2en"}' http://localhost:7000/translate
```

### Example 2: Translating from English to Chinese

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Want to live longer? Play with your grandkids. It’s good for them, too.", "task": "en2zh"}' http://localhost:7000/translate
```

## API Example

### Request

```json
{
  "text": "欧洲经济增长仍面临较大挑战",
  "task": "zh2en"  // Use "zh2en" for Chinese to English, and "en2zh" for English to Chinese translation.
}
```

### Response

For the given request:

```json
{
  "data": {
    "translation": "Economic growth in Europe continues to face significant challenges",
    "info": {}
  }
}
```

### Parameters Description

#### Request Parameters

- `text`: The text you wish to translate.
- `task`: Defines the translation direction. Use "zh2en" for Chinese to English, and "en2zh" for English to Chinese.

#### Response Parameters

- `data`:
  - `translation`: The translated output text.
  - `info`: Additional details about the request, including the model used, original text, etc.
