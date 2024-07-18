import copy
import numpy as np
import os


def save_model_state_dict(model, weight_save_folder: str, weight_save_name: str):
    """保存模型的状态字典到文件

    Args:
        model (nn.Module): 需要确保model具有state_dict()方法
        weight_save_folder (str): 权重文件存储的文件夹路径
        weight_save_name (str): 权重文件名称

    Raises:
        AttributeError: Model state_dict not found.
    """
    print("save model state dict")
    import os
    import torch
    import csv
    from tqdm import tqdm
    import numpy as np

    # 检查模型的state_dict
    if not hasattr(model, "state_dict"):
        msg = "Model state_dict not found."
        raise AttributeError(msg)
    # 创建文件夹
    create_directory(weight_save_folder)
    # 变量
    state_dict = model.state_dict()
    binary_file_name = f"{weight_save_name}.bin"
    csv_file_name = f"{weight_save_name}.csv"
    binary_file_path = os.path.join(weight_save_folder, binary_file_name)
    csv_file_path = os.path.join(weight_save_folder, csv_file_name)
    state_dict_total_size = 0  # bytes
    # 逐 Tensor 存储到文件
    print("'Key', 'Data Type', 'Shape'")
    # 打开二进制文件以写入权重
    with open(binary_file_path, "wb") as bin_file:
        # 打开CSV文件以写入权重信息
        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入CSV文件的头部
            csv_writer.writerow(
                [
                    "Key",
                    "Data Type",
                    "Shape",
                    "Total_Bytes(B)",
                    "Total_Bytes(KB)",
                    "Total_Bytes(MB)",
                    "Acc_Bytes(B)",
                    "Acc_Bytes(KB)",
                    "Acc_Bytes(MB)",
                ]
            )
            # 逐 Tensor 存储到文件
            for key, tensor in tqdm(state_dict.items()):
                # 将权重写入二进制文件
                bin_file.write(tensor.cpu().numpy().tobytes())
                # 写入CSV文件的信息
                total_bytes = tensor.element_size() * tensor.numel()
                state_dict_total_size += total_bytes
                row_data = [
                    key,
                    tensor.dtype,
                    list(tensor.size()),
                    total_bytes,
                    total_bytes / 1024,
                    total_bytes / 1024 / 1024,
                    state_dict_total_size,
                    state_dict_total_size / 1024,
                    state_dict_total_size / 1024 / 1024,
                ]
                csv_writer.writerow(row_data)
                print(str(row_data))
            # 存储最终统计的总大小
            csv_writer.writerow(
                [
                    "total size:",
                    f"{state_dict_total_size}B",
                    f"{state_dict_total_size/1024}KB",
                    f"{state_dict_total_size/1024/1024}MB",
                    f"{state_dict_total_size/1024/1024/1024}GB",
                ]
            )
            # 存储列说明
            csv_writer.writerow(
                [
                    "Column declaration:",
                ]
            )
            csv_writer.writerow(
                [
                    "Key - Implications of model weights",
                    "Data Type - The data type of the model weight",
                    "Shape - The shape of the model weight",
                    "Total_Bytes(B) - The total number of bytes of the model weight",
                    "Total_Bytes(KB) - The total number of bytes of the model weight",
                    "Total_Bytes(MB) - The total number of bytes of the model weight",
                    "Acc_Bytes(B) - The total number of bytes accumulated from the model to the current weight",
                    "Acc_Bytes(KB) - The total number of bytes accumulated from the model to the current weight",
                    "Acc_Bytes(MB) - The total number of bytes accumulated from the model to the current weight",
                ]
            )
    # 提示信息
    print(f"Save model state dict to {binary_file_path}, and its info to {csv_file_path}.")


def save_model_structure(model, model_save_folder, model_save_name="model"):
    import os

    # 获取模型的字符串表示
    model_structure = str(model)
    # 构建模型文件名
    model_save_file_name = model_save_name + ".txt"
    # 构建存储路径
    model_save_path = os.path.join(model_save_folder, model_save_file_name)
    msg = f"save model structure to '{model_save_path}'"
    start_msg = ">" * 6 + msg + ">" * 6
    end_msg = "<" * 6 + msg + "<" * 6
    # 提示信息
    print(start_msg)
    # 将模型结构写入文本文件
    with open(model_save_path, "w") as file:
        file.write(model_structure)
    # 提示信息
    print(end_msg)


def create_directory(path):
    import os

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")


# def save_tensor(save_tensor, tensor_name, save_dir='tensors', show_info=True):
#     # 创建保存目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     tensor = copy.deepcopy(save_tensor).cpu()

#     # 获取tensor信息
#     shape = tensor.shape
#     dtype = tensor.dtype

#     # 构造文件名
#     shape_str = 'x'.join(map(str, shape))
#     dtype_str = str(dtype).split('.')[1]  # 去掉前缀 'torch.'
#     base_filename = f"{tensor_name}_shape-{shape_str}_dtype-{dtype_str}"

#     # 二进制文件名
#     bin_filename = os.path.join(save_dir, f"{base_filename}.bin")

#     # 文本文件名
#     txt_filename = os.path.join(save_dir, f"{base_filename}.txt")

#     # 保存二进制文件
#     with open(bin_filename, 'wb') as bin_file:
#         bin_file.write(tensor.numpy().tobytes())

#     # 保存可视化的文本文件
#     with open(txt_filename, 'w') as txt_file:
#         # 将 tensor 转换为 NumPy 数组并写入文本文件
#         np.savetxt(txt_file, tensor.numpy().reshape(-1), fmt='%10.5f')

#     if show_info:
#         print(f"Tensor binary file saved to {bin_filename}")
#         print(f"Tensor text file saved to {txt_filename}")

def save_tensor(tensor, tensor_name, save_dir='tensors', show_info=True):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将张量移动到CPU并克隆
    tensor = tensor.cpu().clone()

    # 获取tensor信息
    shape = tensor.shape
    dtype = tensor.dtype

    # 构造文件名
    shape_str = 'x'.join(map(str, shape))
    dtype_str = str(dtype).split('.')[1]  # 去掉前缀 'torch.'
    base_filename = f"{tensor_name}_shape-{shape_str}_dtype-{dtype_str}"

    # 二进制文件名
    bin_filename = os.path.join(save_dir, f"{base_filename}.bin")

    # 文本文件名
    txt_filename = os.path.join(save_dir, f"{base_filename}.txt")

    # 将 tensor 转换为 NumPy 数组
    tensor_np = tensor.numpy()

    # 保存二进制文件
    with open(bin_filename, 'wb') as bin_file:
        bin_file.write(tensor_np.tobytes())

    # 保存可视化的文本文件
    np.savetxt(txt_filename, tensor_np.reshape(-1), fmt='%10.5f')

    if show_info:
        print(f"Tensor binary file saved to {bin_filename}")
        print(f"Tensor text file saved to {txt_filename}")