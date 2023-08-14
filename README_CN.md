# ImmersiveL

ImmersiveL是一个框架和模型中心，旨在自由连接世界各地的语言。

目前，ImmersiveL应用程序是基于Deepspeed的中英互译翻译框架。主要结构位于app目录中，由Python 3.8+环境、Flask、Deepspeed和PyTorch组成。

**目前的第一个模型是基于bloomz模型进行训练的，其许可证可以在[这里](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)找到。Apache许可证适用于模型的派生部分和此存储库中的其他源代码文件。**

🌐 **阅读[英文版](README.md)**

## 开始使用

1. **克隆存储库并设置环境**

   首先，将ImmersiveL存储库克隆到您的本地计算机：

   ```bash
   git clone https://github.com/immersive-translate/ImmersiveL.git
   ```

   克隆后，导航到`app`目录并安装`requirements.txt`中列出的所有必要包：

   ```bash
   cd ImmersiveL/app
   pip install -r requirements.txt
   ```

2. **运行应用程序**

   如果您使用的是Linux环境，请使用以下命令使用Deepspeed启动应用程序：

   ```bash
   deepspeed --num_gpus 1 app.py
   ```

   对于Windows用户：

   ```bash
   python app_win.py
   ```

   当您看到类似于`* Running on [IP地址]`的消息，这表明应用程序已成功启动。

## 使用ImmersiveL

应用程序启动并运行后，您可以轻松使用提供的翻译端点。

### 示例1：从中文翻译为英文

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "欧洲经济增长仍面临较大挑战", "task": "zh2en"}' http://localhost:7000/translate
```

### 示例2：从英文翻译为中文

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Want to live longer? Play with your grandkids. It’s good for them, too.", "task": "en2zh"}' http://localhost:7000/translate
```

## API示例

### 请求

```json
{
  "text": "欧洲经济增长仍面临较大挑战",
  "task": "zh2en"  // 使用"zh2en"表示从中文翻译为英文，使用"en2zh"表示从英文翻译为中文。
}
```

### 响应

对于上述请求：

```json
{
  "data": {
    "translation": "欧洲的经济增长仍然面临着重大的挑战",
    "info": {}
  }
}
```

### 参数描述

#### 请求参数

- `text`：待翻译的文本。
- `task`：定义翻译方向。使用"zh2en"从中文翻译为英文，使用"en2zh"从英文翻译为中文。

#### 响应参数

- `data`：
  - `translation`：翻译后的输出文本。
  - `info`：关于请求的额外详细信息，包括所使用的模型、原始文本等。
