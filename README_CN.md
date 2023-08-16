# ImmersiveL

ImmersiveL 是一个旨在连接全球语言的开放自由的框架和模型中心。

目前，ImmersiveL 应用是基于 Deepspeed 的中英双向翻译框架。主要结构位于 `app` 目录中，由 Python 3.8+ 环境、Flask、Deepspeed 和 PyTorch 组成。

**目前的第一个模型是基于 bloomz 模型训练的，其许可证可以在[这里](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)找到。Apache 许可证授权给此仓库中的模型的派生部分和其他源代码文件。**

🌐 **[Read in English](README.md)**

## 开始使用

1. **克隆仓库并设置环境**

   首先，将 ImmersiveL 仓库克隆到本地机器：

   ```bash
   git clone https://github.com/immersive-translate/ImmersiveL.git
   ```

   克隆后，导航到 `app` 目录并安装 `requirements.txt` 中列出的所有必要包：

   ```bash
   cd ImmersiveL/app
   pip install -r requirements.txt
   ```

2. **运行应用**

   如果您使用的是 Linux 环境，请使用以下命令使用 Deepspeed 启动应用程序：

   ```bash
   deepspeed --num_gpus 1 app.py
   ```

   对于 Windows 用户：

   ```bash
   python app_win.py
   ```

   当您看到类似于 `* Running on [IP 地址]` 的消息时，表示应用程序已成功启动。

3. **测试部署**

   现在应用程序正在运行，通过从中文翻译为英文来测试它：

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text": "欧洲经济增长仍面临较大挑战", "task": "zh2en"}' http://localhost:7000/v1/translate
   ```

   如果您得到一个翻译结果，说明您的部署成功。

## API端点

### 1. 沉浸式翻译 (`/v1/immersive_translate`)

#### 请求

- **方法:** POST
- **内容类型:** application/json

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

- `source_lang`: 源语言代码。
- `target_lang`: 目标语言代码。
- `text_list`: 待翻译的文本字符串数组。

#### 响应

对于给定的请求：

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

- `translations`: 包含以下内容的数组：
  - `detected_source_lang`: 已翻译文本的检测语言代码。
  - `text`: 已翻译的文本。

#### 语言代码

- `zh-CN`: 简体中文
- `en`: 英语

### 2. 基本翻译 (`/v1/translate`)

- **方法:** POST
- **内容类型:** application/json

#### 请求参数

- `text`: 您希望翻译的文本。
- `task`: 定义翻译方向。使用 "zh2en" 进行中英翻译，使用 "en2zh" 进行英中翻译。

#### 响应参数

- `data`:
  - `translation`: 已翻译的输出文本。
  - `info`: 有关请求的其他详细信息，包括使用的模型、原始文本等。
