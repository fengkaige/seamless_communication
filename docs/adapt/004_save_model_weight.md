

# 1. 实现原则
- 工具类的函数封装到 : `src/model_weight_save.py`
- 一定要有配置开关，来决定是否存储权重，且开关默认关闭；通过配置的方式打开；
- 存储权重时一定给出提示

# 2. 相关的类和文件
- ModelSaveWeightFlags : 配置是否存储相关位置的模型权重
- src/model_weight_save.py :  存储权重相关的工具

# 3. 存储方法
- 模型权重和golden的存储都使用`main_demo_debug.py`文件作为主入口
- 运行方法：
``` shell
cd seamless_communication/src
python3 main_demo_debug.py
```

# 4. 模型权重存储的流程 - Agent3 OfflineWav2VecBertEncoderAgent 中的 speech_encoder 权重存储的操作示例
1. 实例化ModelSaveWeightFlags类对象，配置权重存储标记; (如果新增一个模型，则配置新增一项)
2. 调用ModelSaveWeightFlags类对象的init_os_env()方法; `seamless_communication/src/main_demo_debug.py`中对应代码:
``` python
# save weight flag by os env.
save_weight_flag = ModelSaveWeightFlags()
save_weight_flag.save_offlineWav2VecBertEncoderAgent = "True"
save_weight_flag.save_vocoderAgent = "False"
save_weight_flag.init_os_env()
```

3. 在具体的模型中判断是否存储该模型的权重； 示例：
``` python
# 代码位置: seamless_communication/src/seamless_communication/streaming/agents/offline_w2v_bert_encoder.py
env_name = "SAVE_OFFLINEWAV2VECBERTENCODERAGENT"
save_offlineWav2VecBertEncoderAgent = os.environ.get(env_name)
if save_offlineWav2VecBertEncoderAgent == ["Flase", "True"][1]:
    # 你的存储权重的代码
    # 建议将权重存储的代码单独写一个函数
    save_weight_of_encode_speech(self.model)
```

4. 实现权重存储的逻辑；示例：
``` python
# 代码位置: seamless_communication/src/seamless_communication/streaming/agents/offline_w2v_bert_encoder.py
def save_weight_of_encode_speech(model):
    """
        调试信息:
            type model : Class seamless_communication.models.unity.model.UnitYModel
            type (model.speech_encoder) : <class 'seamless_communication.models.unity.adaptor_block.UnitYEncoderAdaptor'>

            model.speech_encoder._modules.keys()
                odict_keys(['inner', 'inner_layer_norm', 'proj1', 'activation', 'proj2', 'adaptor_layers', 'layer_norm'])
            model.speech_encoder._modules['inner']._modules.keys()
                odict_keys(['layers', 'layer_norm'])
    """
    from seamless_communication.src.model_weight_save import save_model_state_dict, save_model_structure

    # 提示信息
    print(">" * 12, "save weight of encode_speech", ">" * 12)
    # 构建存储文件夹和存储名称
    weight_save_folder = "./model_weight/Agent3_OfflineWav2VecBertEncoderAgent"
    weight_save_name = "encode_speech_weight"
    # 存储权重
    save_model_state_dict(model.speech_encoder, weight_save_folder, weight_save_name)
    # 存储模型结构
    save_model_structure(model.speech_encoder, weight_save_folder, weight_save_name)
    # 提示信息
    print("<" * 12, "save weight of encode_speech", "<" * 12)
```