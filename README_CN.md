# ImmersiveL

ImmersiveL 是一个连接世界各地语言的框架和模型中心，它是开放且免费的。

目前，ImmersiveL app 是基于 Deepspeed 的翻译框架。主要结构位于 `app` 目录中，由 Python 3.8+、Flask、Deepspeed 和 PyTorch 环境组成。

**目前的第一个模型是基于bloomz模型进行训练的，其许可证可以在[这里](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)找到。Apache许可证授权给该模型的派生部分以及此仓库中的其他源代码文件。**

🌐 **阅读 [English (英文)](README.md) 版本**

## 入门指南

1. **克隆仓库**

   首先，将 ImmersiveL 仓库clone到本地计算机。

   ```bash
   git clone https://github.com/immersive-translate/ImmersiveL.git
   ```

2. **导航至 App 目录，并安装依赖**

   clone完成后，移动到 `app` 目录。然后安装 `requirements.txt` 中列出的所有必需的包。

   ```bash
   cd ImmersiveL/app
   pip install -r requirements.txt
   ```

3. **下载模型**

   模型可以在[此 Huggingface 链接](https://huggingface.co/funstoryai/immersiveL-exp/tree/main)中找到。下载 "Files and versions" 部分下的文件，并将它们放入 `app/model` 目录。

4. **运行应用程序**

   使用以下命令使用 Deepspeed 启动应用程序：

   ```bash
   deepspeed --num_gpus 1 app.py
   ```

   当你看到类似于 `* Running on [IP 地址]` 的消息时，表示应用程序已成功启动。

## 使用 ImmersiveL

应用程序启动并运行后，您可以轻松使用提供的翻译端点。

### 示例1：从中文翻译为英文

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "欧洲经济增长仍面临较大挑战", "task": "zh2en"}' http://localhost:7000/translate
```

### 示例2：从英文翻译为中文

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Want to live longer? Play with your grandkids. It’s good for them, too.", "task": "en2zh"}' http://localhost:7000/translate
```
