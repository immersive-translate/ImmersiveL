# 开发者测试文档

## 前言

测试文件夹`test`仅用于开发者测试

---

## 任务一: 设置本地代理

为了在本地进行测试，我们使用SSH隧道创建一个代理连接到远程服务器。

```bash
# 本地代理设置，根据情况修改
ssh -CNg -L 7000:127.0.0.1:7000 root@192.168.1.2 -p 12345
```

### 任务二: 单一请求测试

在进行并发测试之前，首先确保API端点正常工作。

```bash
curl --location 'http://127.0.0.1:7000/v1/immersive_translate' \
--header 'Content-Type: application/json' \
--data '{
    "source_lang": "zh-CN",
    "target_lang": "en",
    "text_list": [
        "这是一个测试句子",
        "欧洲经济增长仍面临较大挑战",
        "Vue 是一套用于构建用户界面的渐进式框架"
    ]
}'
```

正常响应应该是一个包含翻译的JSON结果。

### 任务三: 并发性能测试

1. **定义bash函数**:

    在test文件夹下创建名为`output`的文件夹，为了执行并发请求并保存每个请求的结果，我们定义一个bash函数：

    ```bash
    do_curl() {
        curl -X POST -H 'Content-Type: application/json' -d @test/data.json "$1" -o "./test/output/output${2}.txt"
    }
    export -f do_curl
    ```

2. **启动并发测试**:

    使用`parallel`工具进行并发请求：

    ```bash
    seq 50 | parallel -j 5 do_curl 'http://127.0.0.1:7000/v1/immersive_translate' {}
    ```

    这将发出50个请求，每次5个并发。

3. **分析结果**:

    在`test/output/`目录下，您可以看到每个请求的输出文件（例如`output1.txt`、`output2.txt`等）。这些文件应包含API的响应。
