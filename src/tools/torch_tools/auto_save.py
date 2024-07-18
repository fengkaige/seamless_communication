import torch
from .model_weight_save import save_tensor


"""
# Golden 文件格式描述

整体都有2个格式的文件，一个是bin，一个是txt
bin是二进制格式
txt是可视化文本格式(十进制)

# 带有Index 的 Golden 命名规则描述

## 输入命名规则


index_{full_layer_name}_input_{i}_shape-{shape}_dtype-{dtype}.bin

index_{full_layer_name}_input_{i}_shape-{shape}_dtype-{dtype}.txt

index : 不用太关注，存储时从0递增的，仅作顺序参考; 注意详细算子Golden里面是一层所有的算子Golden存储完成，才存储整体的输入输出，所以整个层的输入输出的golden的index偏大。
{full_layer_name} : 示例: speech_encoder.inner.layers.0; 可以和权重的csv文件对应一下，基本能对上。也可以和pytorch或者onnx的模型结构对应。
input_{i} : 标识第几个input
shape-{shape} : 描述形状信息
dtype-{dtype} : 描述数据类型信息


## 输出命名规则


index_{full_layer_name}_output_{i}_shape-{shape}_dtype-{dtype}.bin

index_{full_layer_name}_output_{i}_shape-{shape}_dtype-{dtype}.txt
"""


# 钩子函数
def save_activation(name, path):
    def hook(model, inputs, output):
        # 保存输入
        for i, input in enumerate(inputs):
            if isinstance(input, torch.Tensor):
                # shape_str = 'x'.join(map(str, input.shape))
                # dtype_str = str(input.dtype).split('.')[1]  # 去掉前缀 'torch.'
                # input_info = f"{name}_input_{i}_shape_{shape_str}_dtype_{dtype_str}"
                # input_full_path = f"{path}/{input_info}.pt"
                # print("auto save", input_full_path)
                # torch.save(input, input_full_path)
                save_tensor(
                    input,
                    tensor_name=name + f"_input_{i}",
                    save_dir=path,
                    show_info=True,
                )

        # 保存输出
        if isinstance(output, torch.Tensor):
            output_info = (
                f"{path}/{name}_output_shape_{output.shape}_dtype_{output.dtype}.pt"
            )
            torch.save(output, output_info)
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    # shape_str = 'x'.join(map(str, out.shape))
                    # dtype_str = str(out.dtype).split('.')[1]  # 去掉前缀 'torch.'
                    # output_info = f"{path}/{name}_output_{i}_shape_{shape_str}_dtype_{dtype_str}.pt"
                    # print("auto save", output_info)
                    # torch.save(out, output_info)
                    save_tensor(
                        out,
                        tensor_name=name + f"_output_{i}",
                        save_dir=path,
                        show_info=True,
                    )

    return hook


# 自动注册钩子函数
def register_save_io_hooks(module, path, prefix=""):
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        submodule.register_forward_hook(save_activation(full_name, path))
        register_save_io_hooks(submodule, path, full_name)


# 自动注册钩子函数 - 带有全局索引
global save_index
save_index = 0


def register_save_io_hooks_with_index(module, path, prefix=""):
    """注册存储模型中各个算子的输入输出的Golden函数
    该函数在模型的forward调用结束之后调用

    Args:
        module (torch.nn.Module): torch.nn.Module格式的模型
        path (str): 存储路径
        prefix (str, optional): 存储Golden的前缀. Defaults to "".

    Example:
        save_path = "./test_auto_save"
        # 创建路径
        create_directory(save_path)
        ### 注册存储输入输出的函数 - 文件名带有全局index
        register_save_io_hooks_with_index(model, path = save_path)
        # 推理模型 - 推理过程中会调用到注册的函数来存储输入和输出
        model(*inputs)
    """
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        submodule.register_forward_hook(save_activation_with_index(full_name, path))
        register_save_io_hooks_with_index(submodule, path, full_name)


# 钩子函数 - 带有全局索引
def save_activation_with_index(name, path):
    def hook(model, inputs, output):
        global save_index
        # 保存输入
        for i, input in enumerate(inputs):
            if isinstance(input, torch.Tensor):
                save_tensor(
                    input,
                    tensor_name=str(save_index) + "_" + name + f"_input_{i}",
                    save_dir=path,
                    show_info=True,
                )

        # 保存输出
        if isinstance(output, torch.Tensor):
            save_tensor(
                        output,
                        tensor_name=str(save_index) + "_" + name + f"_output_0",
                        save_dir=path,
                        show_info=True,
            )
            # torch.save(output, output_info)
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    save_tensor(
                        out,
                        tensor_name=str(save_index) + "_" + name + f"_output_{i}",
                        save_dir=path,
                        show_info=True,
                    )
        else:
            print("[WARN]output type is '{type(output)}'")
        save_index += 1

    return hook
