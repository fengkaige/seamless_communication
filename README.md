# Main Demo 运行方法
1. 启动虚拟环境 (确保环境已安装)
2. 执行`main_demo`
``` shell
cd src
python3.11 main_demo.py
```

# 文档
- 模型概述 [000_seamless](./docs/adapt/000_seamless.md)
- 搭建环境 [001_env](./docs/adapt/001_env.md)
- 运行main_demo [002_main_demo](./docs/adapt/002_main_demo.md)
- main_demo数据流 [003_data_stream](./docs/adapt/003_data_stream.md)
- 存储权重说明文档 [004_model_weight](./docs/adapt/004_save_model_weight.md)

# 注意事项
## 1. 源码修改备注
下面2个文件中的部分代码被修改成了固定路径，用来加载模型
`src/seamless_communication/models/unity/char_tokenizer.py`
`venv_seamless/lib/python3.11/site-packages/fairseq2/models/utils/generic_loaders.py`文件中`TokenizerLoaderBase`类

详见 [002_main_demo](./docs/adapt/002_main_demo.md)

