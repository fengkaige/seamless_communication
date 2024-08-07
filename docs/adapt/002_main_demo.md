> *该文件确保main_demo可以成功运行*

# 1. Main Demo 运行方法
## 1.1 搭建环境
[搭建环境参考](./001_env.md)
## 1.2 运行方法
``` shell
source your_venv/bin/activate # 激活你的虚拟环境
python3 main_demo.py # 执行脚本
```

---

# 2. Main Demo 支持过程记录
# 2.1 step 1 : 模型依赖
- 运行卡在`Downloading the tokenizer of seamless_streaming_unity...`阶段
- 原因是缺少模型文件 -> 模型文件下载 [huggingface](https://huggingface.co/facebook/seamless-streaming/tree/main)
## 2.1.1 模型依赖修改概述
- 为什么修改
	- 原因1：网络问题，无法在运行代码时下载模型；
	- 原因2：该工程和依赖库不支持传入模型文件路径离线运行
- 怎么修改 - 模型加载阶段的修改主要涉及2个文件:
	- 文件1: `venv_seamless/lib/python3.11/site-packages/fairseq2/models/utils/generic_loaders.py`文件中`TokenizerLoaderBase`类和`ModelLoader`类
	- 文件2: `seamless_communication/src/seamless_communication/models/unity/char_tokenizer.py`文件中`ModelLoader`类
	- 共涉及5个模型加载的修改, 模型名称:
		- seamless_streaming_monotonic_decoder.pt
		- seamless_streaming_unity.pt
		- spm_char_lang38_tc.model
		- tokenizer.model
		- vocoder_v2.pt
- 修改内容 - 替换模型加载路径后的文件内容
	- 第一个文件:
		- 文件路径(在虚拟环境中): `your_python_venv/lib/python3.11/site-packages/fairseq2/models/utils/generic_loaders.py`
		- 替换后的文件内容 : [generic_loaders.py](./code_back_up/generic_loaders.py)
	- 第二个文件:
		- 文件路径(在工程中): `seamless_communication/src/seamless_communication/models/unity/char_tokenizer.py`
		- 替换后的文件内容 : [char_tokenizer.py](./code_back_up/char_tokenizer.py)
	- **注意: 上述2个文件里面`path_zoo`字典里的键值对中的`value`值需要替换为你本地实际模型文件路径**
- 下面`2.1.2`、`2.1.3`、`2.1.4`章节不用过度关注

## 2.1.2 **第1个加载项**的修改过程
- text_tokenizer = load_unity_text_tokenizer(args.unity_model_name)
	- pos：`seamless_communication/src/seamless_communication/streaming/agents/unity_pipeline.py`
	- value: `args.unity_model_name :  seamless_streaming_unity`
- -> load_unity_text_tokenizer = **NllbTokenizerLoader**(asset_store, download_manager)
- -> from fairseq2.models.nllb.loader import **NllbTokenizerLoader**
- -> **NllbTokenizerLoader**::`__call__`
	- `NllbTokenizerLoader(TokenizerLoaderBase[NllbTokenizer])`
	- file: `venv_seamless/lib/python3.11/site-packages/fairseq2/models/nllb/loader.py`
- -> **TokenizerLoaderBase**::`__call__`
- -> from fairseq2.models.utils import ..., **TokenizerLoaderBase**
- -> **TokenizerLoaderBase**::`__call__`
	- file: venv_seamless/lib/python3.11/site-packages/fairseq2/models/utils/generic_loaders.py
	- func: `__call__`
		```
		def __call__(
		        self,
		        model_name_or_card: Union[str, AssetCard],
		        *,
		        force: bool = False,
		        progress: bool = True,
		    ) -> TokenizerT:
		```
	- **第一个加载项**
	- 手动下载: `https://huggingface.co/facebook/seamless-m4t-large/resolve/main/tokenizer.model`
	- 手动替换path: `path = self.download_manager.download_tokenizer ... `

## 2.1.3 **第二个加载项**的修改过程
- 位置
``` shell
File "/home/fengkaige/codespace/seamless/seamless_communication/src/seamless_communication/streaming/agents/unity_pipeline.py", line 132, in load_model
unity_model = load_unity_model(asset_card, device=args.device, dtype=args.dtype)
		  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/fengkaige/codespace/seamless/venv_seamless/lib/python3.11/site-packages/fairseq2/models/utils/generic_loaders.py", line 225, in __call__
path = self.download_manager.download_checkpoint(
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
- 替换内容: 替换`path`变量为绝对路径

## 2.1.4 第3个加载项的修改过程
- 位置
``` shell
  File "/home/fengkaige/codespace/seamless/seamless_communication/src/seamless_communication/models/unity/t2u_builder.py", line 593, in build_decoder_frontend
    char_tokenizer = load_unity_char_tokenizer(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fengkaige/codespace/seamless/seamless_communication/src/seamless_communication/models/unity/char_tokenizer.py", line 106, in __call__
    pathname = self.download_manager.download_tokenizer(
```
- 替换内容: pathname
- 相关模型和链接:
	https://huggingface.co/facebook/seamless-m4t-v2-large/resolve/main/spm_char_lang38_tc.model
	seamlessM4T_v2_large

# 2.2 step 2. 执行main_demo
完成 step 1 main_demo 在 8G Tesla P4 GPU跑通

# 2.3 step 3. 目录结构
main_demo 跑通时的目录结构

``` shell
seamless
├── seamless_communication # seamless工程
│   ├── 23-11_SEAMLESS_BlogHero_11.17.jpg
│   ├── ACCEPTABLE_USE_POLICY
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── demo
│   ├── dev_requirements.txt
│   ├── docs
│   ├── ggml
│   ├── LICENSE
│   ├── MIT_LICENSE
│   ├── pyproject.toml
│   ├── README.md
│   ├── SEAMLESS_LICENSE
│   ├── seamlessM4T.png
│   ├── Seamless_Tutorial.ipynb
│   ├── setup.py
│   ├── src
│   │   ├── input # new
│   │   │   └── reading.wav
│   │   ├── main_demo.py # new
│   │   ├── output # new
│   │   │   ├── reading.wav
│   │   │   ├── wave0.png
│   │   │   ├── wave1.png
│   │   │   ├── wave2.png
│   │   │   └── wave3.png
│   │   └── seamless_communication
│   └── tests
├── seamless-streaming-card
│   ├── gitattributes
│   ├── README.md
│   ├── seamless_streaming_monotonic_decoder.pt
│   ├── seamless_streaming_unity.pt
│   ├── spm_char_lang38_tc.model
│   ├── streaming_arch.png
│   ├── tokenizer.model
│   └── vocoder_v2.pt
└── venv_seamless
```