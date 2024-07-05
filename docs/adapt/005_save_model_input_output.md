整体和权重存储类似: `004_save_model_weight.md`

# 1. 相关类和函数
- 存储Tensor的函数: src/model_weight_save.py 中 save_tensor函数

# 2. 模型输入输出存储的流程 - Agent3 OfflineWav2VecBertEncoderAgent 中的 speech_encoder 权重存储的操作示例
- 整体和存储权重的过程是类似的；
- 具体的存储逻辑:
    - 文件: seamless_communication/src/seamless_communication/streaming/agents/offline_w2v_bert_encoder.py
    - 函数: save_input_output_speech_encoder