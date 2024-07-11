import bz2
import torch
import base64
import ctypes
import os
import sys
import traceback

from torch.nn.parameter import Parameter
from transformers.utils import logging

from typing import List

logger = logging.get_logger(__name__)

try:
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up

    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code
            self._function_names = function_names
            self._cmodule = LazyKernelCModule(self.code)

            for name in self._function_names:
                setattr(self, name, KernelFunction(self._cmodule, name))


    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4WeightCompression",
            "int4WeightExtractionFloat",
            "int4WeightExtractionHalf",
            "int4WeightExtractionBfloat16",
            "int8WeightExtractionFloat",
            "int8WeightExtractionHalf",
            "int8WeightExtractionBfloat16",
            "int4WeightExtractionTransFloat",
            "int4WeightExtractionTransHalf",
            "int4WeightExtractionTransBfloat16",
            "int8WeightExtractionTransFloat",
            "int8WeightExtractionTransHalf",
            "int8WeightExtractionTransBfloat16",
            "int4WeightPerChannelLdkMultiplicationFloat_2",
            "int4WeightPerChannelLdkMultiplicationHalf_2",
            "int4WeightPerChannelLdkMultiplicationBfloat16_2",
            "int8WeightPerChannelLdkMultiplicationFloat_2",
            "int8WeightPerChannelLdkMultiplicationHalf_2",
            "int8WeightPerChannelLdkMultiplicationBfloat16_2",
            "int4GemmHalf",
            "int4GemmBfloat16",
            "int4GemmFloat",
            "int8GemmHalf",
            "int8GemmBfloat16",
            "int8GemmFloat",
        ],
    )
except Exception as exception:
    kernels = None
    logger.warning("Failed to load cpm_kernels:" + str(exception))


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        quant_w: torch.Tensor,
        scale_w: torch.Tensor,
        weight_bit_width,
    ):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        if inp.size(0) > 8:
            weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
            ctx.weight_shape = weight.size()
            output = inp.mm(weight.t())

        else:
            output = quant_gemv(inp, quant_w, scale_w)
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return (
            grad_input.view(ctx.inp_shape),
            grad_weight.view(ctx.weight_shape),
            None,
            None,
        )


default_cpu_kernel_code_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "quantization_kernels.c"
)
default_cpu_kernel_code = "QlpoOTFBWSZTWXLbSoQAAgzbgERwQXxmTwAAr/ff3kABt0Q2oRVT0hpo9RtEAAAAyBEiSQ9EGjQGQAAAwANGhowjJoNGmgMEUplMTNSMJ5TQaDJpsoMyRMj8P4mZzFSVVwqSXG8GG7MlVwiToYEQwVD7noBxMhNfkeZYtYFtbgOBUSIGtIQjhNHCEnPJsadhb3yBmRIOD3TeAtNLSaU5GgvKUBWSNuuOIHmVt0YhW6rsmDMDUjeUJGJ64R1Jm5lrh0Aa0tKjhFwPdWcGogxLDSXPWQUWTM8Sd3Qz1HMYNxx3HMeiNqNo4jeRDEfZ3gUSHIcU/heomq0vEzL1Msz5KKGxH8FrNOYw3KaxdqaEmNHYMxJFgQbR0DyRknL2L4kwUSxKRdhjRpEtUqilVfggFL1klaMS3PPRDfNqbBOPWO7m4JTVGhS9QTBDDJaEbLbrUQNB+IpJSKQbG5SZZ5gkwJEhJ3aYKJipZ/i7kinChIOW2lQg"
default_cpu_parallel_kernel_code_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "quantization_kernels_parallel.c"
)
default_cpu_parallel_kernel_code = "QlpoOTFBWSZTWUzax5EAALXbgERwSX1mTwAAr/ff3kACNyXUbZYwBpoaNGIyAaADQwRSaVP9QoMg0A2oAPU0AEUkU9GaaKMaQB6gA09T1ARRKnpk0niaJkaaNDJ6g0DTIKVKfZ/g6v1Kem5LJLa0WmkukkuCIHUqWbtJGJMsCSQFiPEIYHgBIZDzR8R6REbYxIqD2Cu7lMkFoPu6LmHeOAy0GF83Tc40jgmTs4HnCe60QfJa2bDBZ0Y1lhgbiZjW8SNsAKCk42UOEdjWN3KoiCIYeQUCCKWIyHewhtSoInLKSG22l4jKM2ZDCVKtBm3OTYBl3jsVqMImtj7PQw7xKxLXQzwgJaPPgW1fRhrvPJICl4YFDYfNbkbBh5JDgrazFml50xEQQwQUjxNwE0IDSofLzSg7UNVKn+Rr1KErzBHUxBqdHRlXzqYsIa5K9Y0UuE2ugw3g5KYofm7AaGNTzJSMhcchhxdaU4JZ0F1UNgQ8XcGDguypqYza8yFaEoGgNRcLej+g2t0feGKFE5OY2PFluQ3q4HgycxlfvzHqo0KcM0JI8OKXtzayJFgsqC1NdUQVu8rChnA6FO3MFyGOoC9KO8ITPpYM5pRqTlczFkLES/4u5IpwoSCZtY8i"


class CPUKernel:
    def __init__(
        self,
        kernel_file="",
        source_code=default_cpu_kernel_code_path,
        compile_parallel_kernel=None,
        parallel_num=None,
    ):
        self.load = False
        self.int8WeightExtractionFloat = None
        self.int4WeightExtractionFloat = None
        self.int4WeightCompression = None
        self.SetNumThreads = lambda x: x

        try:
            if not os.path.exists(default_cpu_kernel_code_path):
                with open(default_cpu_kernel_code_path, "w", encoding="utf-8") as file:
                    code = default_cpu_kernel_code
                    cpu_quantization_code = bz2.decompress(
                        base64.b64decode(code)
                    ).decode()
                    file.write(cpu_quantization_code)

            if not os.path.exists(default_cpu_parallel_kernel_code_path):
                with open(
                    default_cpu_parallel_kernel_code_path, "w", encoding="utf-8"
                ) as file:
                    code = default_cpu_parallel_kernel_code
                    cpu_quantization_code = bz2.decompress(
                        base64.b64decode(code)
                    ).decode()
                    file.write(cpu_quantization_code)

        except Exception:
            logger.warning("Error when generating default cpu kernel code.")

        if compile_parallel_kernel is None:
            compile_parallel_kernel = bool(int(os.cpu_count()) >= 4)

        if compile_parallel_kernel and source_code == default_cpu_kernel_code_path:
            source_code = default_cpu_parallel_kernel_code_path

        kernels = None

        if (not kernel_file) or (not os.path.exists(kernel_file)):
            try:
                if os.path.exists(source_code):
                    kernel_file = source_code[:-2] + ".so"

                    if compile_parallel_kernel:
                        if sys.platform != "darwin":
                            compile_command = "gcc -O3 -fPIC -pthread -fopenmp -std=c99 {} -shared -o {}".format(
                                source_code, kernel_file
                            )
                        else:
                            compile_command = "clang -O3 -fPIC -pthread -Xclang -fopenmp -lomp -std=c99 {} -shared -o {}".format(
                                source_code, kernel_file
                            )
                        exit_state = os.system(compile_command)
                        if not exit_state:
                            try:
                                kernels = ctypes.cdll.LoadLibrary(kernel_file)
                            except:
                                logger.warning(
                                    f"Load parallel cpu kernel failed {kernel_file}: {traceback.format_exc()}"
                                )
                        else:
                            logger.warning(
                                f"Compile parallel cpu kernel {compile_command} failed."
                            )

                        if kernels is None:  # adjust config, use default cpu kernel
                            compile_parallel_kernel = False
                            source_code = default_cpu_kernel_code_path
                            kernel_file = source_code[:-2] + ".so"

                    if kernels is None:
                        compile_command = (
                            "gcc -O3 -fPIC -std=c99 {} -shared -o {}".format(
                                source_code, kernel_file
                            )
                        )
                        exit_state = os.system(compile_command)
                        if not exit_state:
                            try:
                                kernels = ctypes.cdll.LoadLibrary(kernel_file)
                            except:
                                logger.warning(
                                    f"Load cpu kernel {kernel_file} failed: {traceback.format_exc()}"
                                )
                        else:
                            logger.warning(
                                f"Compile cpu kernel {compile_command} failed."
                            )
                else:
                    logger.warning("Kernel source code not found.")
                    return
            except:
                logger.warning(f"Failed to build cpu kernel: {traceback.format_exc()}")
                return
        else:
            try:
                kernels = ctypes.cdll.LoadLibrary(kernel_file)
            except:
                logger.warning(
                    f"Load custom cpu kernel {kernel_file} failed: {traceback.format_exc()}"
                )

        if kernels is not None:
            self.int8WeightExtractionFloat = kernels.extract_int8_weight_to_float
            self.int4WeightExtractionFloat = kernels.extract_int4_weight_to_float
            self.int4WeightCompression = kernels.compress_int4_weight
            if compile_parallel_kernel:
                try:
                    self.SetNumThreads = kernels.set_num_threads
                except:
                    logger.warning("No set_num_threads() found in kernel.")
            self.load = True

        if compile_parallel_kernel:
            if parallel_num is None:
                parallel_num = max(os.cpu_count(), 1)
            self.SetNumThreads(parallel_num)

        self.parallel_num = parallel_num


cpu_kernels = CPUKernel()


def extract_weight_to_float(
    weight: torch.Tensor,
    scale_list: torch.Tensor,
    source_bit_width: int,
    quantization_cache=None,
):
    """extract weight on cpu to float32"""
    if source_bit_width == 8:
        func = cpu_kernels.int8WeightExtractionFloat
    elif source_bit_width == 4:
        func = cpu_kernels.int4WeightExtractionFloat
    else:
        assert False, "Unsupported bit-width"

    n, m = weight.size(0), weight.size(1)

    if quantization_cache is not None:
        out = quantization_cache
        func(
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(scale_list.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
        )
        return out.tensor
    else:
        out = torch.empty(
            n, m * (8 // source_bit_width), dtype=torch.float, device="cpu"
        )
        func(
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(scale_list.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
        )
        return out


class W8A16LinearCPU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        quant_w: torch.Tensor,
        scale_w: torch.Tensor,
        weight_bit_width,
        quantization_cache=None,
    ):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_float(
            quant_w, scale_w, weight_bit_width, quantization_cache=quantization_cache
        )
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_float(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return (
            grad_input.view(ctx.inp_shape),
            grad_weight.view(ctx.weight_shape),
            None,
            None,
        )


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with weight.device:
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8)
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        kernels.int4WeightCompression(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


def compress_int4_weight_cpu(weight: torch.Tensor):  # (n, m)
    n, m = weight.size(0), weight.size(1)
    assert m % 2 == 0
    m = m // 2
    out = torch.empty(n, m, dtype=torch.int8)

    cpu_kernels.int4WeightCompression(
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int32(n),
        ctypes.c_int32(m),
    )
    return out


def extract_weight_to_half(
    weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int
):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf  # int4 - > half
    else:
        assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(
            n, m * (8 // source_bit_width), dtype=torch.half, device="cuda"
        )
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


def extract_weight_to_bfloat16(
    weight: torch.Tensor,
    scale_list: torch.Tensor,
    source_bit_width: int,
    trans: bool = False,
):
    if trans:
        if source_bit_width == 8:
            func = kernels.int8WeightExtractionTransBfloat16
        elif source_bit_width == 4:
            func = kernels.int4WeightExtractionTransBfloat16
        else:
            assert False, "Unsupported bit-width"
    else:
        if source_bit_width == 8:
            func = kernels.int8WeightExtractionBfloat16
        elif source_bit_width == 4:
            func = kernels.int4WeightExtractionBfloat16
        else:
            assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(
            n, m * (8 // source_bit_width), dtype=torch.bfloat16, device="cuda"
        )
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


def quant_gemv(input: torch.Tensor, weight: torch.Tensor, scale_list: torch.Tensor):
    #    for T = half4
    #    weight is int8 [n, k] row-major
    #    input is [m, k]
    #    scale_list is [n] for per_channel quantization.
    #    output is [m, n]
    #    each thread deals with at least m * 4 cols (k)
    #    each block deals with nPerThread m * rows (n)
    #    assume n % nPerThread == 0 && k % 4 == 0
    #    grid(n/nPerThread)
    assert weight.dtype == torch.int8, "weight type should be int8"
    assert weight.dim() == 2, "weight dim should be 2"
    if weight.size(1) == input.size(1):
        bit_width = 8
    elif weight.size(1) * 2 == input.size(1):
        bit_width = 4
    else:
        assert False, "unknown quant bit width"

    nPerThread = 2

    if input.dtype == torch.float:
        input_type = "Float"
    elif input.dtype == torch.float16:
        input_type = "Half"
    elif input.dtype == torch.bfloat16:
        input_type = "Bfloat16"
    else:
        assert False, f"unsupport input type: {input.dtype}"

    func_name = (
        f"int{bit_width}WeightPerChannelLdkMultiplication{input_type}_{nPerThread}"
    )
    func = getattr(kernels, func_name, None)
    if func == None:
        assert False, f"{func_name} is not supported"
    with torch.cuda.device(weight.device):
        m, n, k = input.size(0), weight.size(0), input.size(1)
        out = torch.empty(m, n, dtype=input.dtype, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n // nPerThread, 1, 1)
        blockDim = [1, 1, 1]
        if k > 10000:
            blockDim[0] = 256
        elif k > 2000:
            blockDim[0] = 128
        else:
            blockDim[0] = 64
        while blockDim[0] * 4 > k:
            blockDim[0] //= 2
        blockDim[0] = (blockDim[0] + 31) // 32 * 32
        blockDim = tuple(blockDim)

        shm_size = blockDim[0] * nPerThread * 4

        func(
            gridDim,
            blockDim,
            shm_size,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(input.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(m),
                ctypes.c_int32(k),
            ],
        )
        return out


def quant_gemm(input: torch.Tensor, weight: torch.Tensor, scale_list: torch.Tensor):
    assert weight.dtype == torch.int8, "weight type should be int8"
    assert weight.dim() == 2, "weight dim should be 2"
    if weight.size(1) == input.size(1):
        bit_width = 8
    elif weight.size(1) * 2 == input.size(1):
        bit_width = 4
    else:
        assert False, "unknown quant bit width"

    if input.dtype == torch.float:
        input_type = "Float"
    elif input.dtype == torch.float16:
        input_type = "Half"
    elif input.dtype == torch.bfloat16:
        input_type = "Bfloat16"
    else:
        assert False, f"unsupport input type: {input.dtype}"

    func_name = f"int{bit_width}Gemm{input_type}"
    func = getattr(kernels, func_name, None)
    if func == None:
        assert False, f"{func_name} is not supported"
    with torch.cuda.device(weight.device):
        n, m, k = input.size(0), input.size(1), weight.size(0)
        out = torch.empty(n, k, dtype=input.dtype, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = ((n - 1) // 32 + 1, (k - 1) // 32 + 1, 1)
        blockDim = (32, 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(input.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
                ctypes.c_int32(k),
            ],
        )
        return out


def python_compress_int4_weight(weight):
    n, m = weight.size(0), weight.size(1)  # (4608,4096)
    assert m % 2 == 0
    m = m // 2
    out = torch.zeros(n, m, dtype=torch.int8)
    for i in range(n):
        for j in range(m):
            out_byte = (weight[i, j * 2] << 4) & 0xF0
            out_byte |= weight[i, j * 2 + 1] & 0x0F
            out[i, j] = out_byte
    return out


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        weight_bit_width: int,
        weight,
        bias=None,
        device="cpu",
        dtype=None,
        empty_init=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.weight_bit_width = weight_bit_width

        shape = weight.shape

        if weight is None or empty_init:
            self.weight = torch.empty(
                shape[0],
                shape[1] * weight_bit_width // 8,
                dtype=torch.int8,
                device=device,
            )
            self.weight_scale = torch.empty(shape[0], dtype=dtype, device=device)
        else:
            weight = weight
            self.weight_scale = weight.abs().max(dim=-1).values / (
                (2 ** (weight_bit_width - 1)) - 1
            )
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(
                torch.int8
            )
            if weight_bit_width == 4:

                if device.type == "cpu":
                    self.weight = compress_int4_weight_cpu(self.weight)
                else:
                    self.weight = compress_int4_weight(self.weight)

                # if torch.cuda.is_available():
                #     self.weight = compress_int4_weight(self.weight)
                # else:
                #     self.weight = compress_int4_weight_cpu(self.weight)

        self.weight = Parameter(self.weight.to(device), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(device), requires_grad=False)
        self.bias = (
            Parameter(bias.to(device), requires_grad=False)
            if bias is not None
            else None
        )

    def forward(self, input):
        if self.weight.device == torch.device("cpu"):
            output = W8A16LinearCPU.apply(
                input, self.weight, self.weight_scale, self.weight_bit_width
            )
        else:
            output = W8A16Linear.apply(
                input, self.weight, self.weight_scale, self.weight_bit_width
            )
        if self.bias is not None:
            output = output + self.bias
        return output


def quantize(
    model,
    weight_bit_width,
    empty_init=False,
    device=None,
    model_type=None,
    lm_hea_quant=False,
):
    """Replace fp16 linear with quantized linear"""
    if lm_hea_quant:
        if model_type in ["chatglm2", "chatglm3", "codegeex2"]:
            model.transformer.output_layer = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.transformer.output_layer.weight,
                bias=model.transformer.output_layer.bias,
                dtype=model.transformer.output_layer.weight.dtype,
                device=(
                    model.transformer.output_layer.weight.device
                    if device is None
                    else device
                ),
                empty_init=empty_init,
            )

        elif model_type in ["llama", "aquila", "openchat"]:
            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight,
                bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )
        elif model_type in ["baichuan"]:
            model.lm_head.weight.data = torch.nn.functional.normalize(
                model.lm_head.weight
            )
            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight.to(device),
                # bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )

        elif model_type in ["qwen"]:
            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight.to(device),
                bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )

        else:
            pass
    else:
        if model_type in ["chatglm2", "chatglm3", "codegeex2"]:
            for layer in model.transformer.encoder.layers:
                layer.self_attention.query_key_value = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    # weight=layer.self_attention.query_key_value.weight.to(torch.cuda.current_device()),
                    weight=layer.self_attention.query_key_value.weight,
                    bias=layer.self_attention.query_key_value.bias,
                    dtype=layer.self_attention.query_key_value.weight.dtype,
                    device=(
                        layer.self_attention.query_key_value.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.self_attention.dense = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attention.dense.weight,
                    # weight=layer.self_attention.dense.weight.to(torch.cuda.current_device()),
                    bias=layer.self_attention.dense.bias,
                    dtype=layer.self_attention.dense.weight.dtype,
                    device=(
                        layer.self_attention.dense.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.dense_h_to_4h = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.dense_h_to_4h.weight,
                    # weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
                    bias=layer.mlp.dense_h_to_4h.bias,
                    dtype=layer.mlp.dense_h_to_4h.weight.dtype,
                    device=(
                        layer.mlp.dense_h_to_4h.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.dense_4h_to_h = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.dense_4h_to_h.weight,
                    # weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
                    bias=layer.mlp.dense_4h_to_h.bias,
                    dtype=layer.mlp.dense_4h_to_h.weight.dtype,
                    device=(
                        layer.mlp.dense_4h_to_h.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )

            model.transformer.output_layer = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.transformer.output_layer.weight,
                bias=model.transformer.output_layer.bias,
                dtype=model.transformer.output_layer.weight.dtype,
                device=(
                    model.transformer.output_layer.weight.device
                    if device is None
                    else device
                ),
                empty_init=empty_init,
            )

        elif model_type in ["llama", "aquila", "openchat"]:

            for layer in model.model.layers:
                layer.self_attn.q_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.q_proj.weight,
                    bias=layer.self_attn.q_proj.bias,
                    dtype=layer.self_attn.q_proj.weight.dtype,
                    device=(
                        layer.self_attn.q_proj.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.self_attn.k_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.k_proj.weight,
                    bias=layer.self_attn.k_proj.bias,
                    dtype=layer.self_attn.k_proj.weight.dtype,
                    device=(
                        layer.self_attn.k_proj.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.self_attn.v_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.v_proj.weight,
                    bias=layer.self_attn.v_proj.bias,
                    dtype=layer.self_attn.v_proj.weight.dtype,
                    device=(
                        layer.self_attn.v_proj.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.self_attn.o_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.o_proj.weight,
                    bias=layer.self_attn.o_proj.bias,
                    dtype=layer.self_attn.o_proj.weight.dtype,
                    device=(
                        layer.self_attn.o_proj.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.gate_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.gate_proj.weight,
                    bias=layer.mlp.gate_proj.bias,
                    dtype=layer.mlp.gate_proj.weight.dtype,
                    device=(
                        layer.mlp.gate_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.down_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.down_proj.weight,
                    bias=layer.mlp.down_proj.bias,
                    dtype=layer.mlp.down_proj.weight.dtype,
                    device=(
                        layer.mlp.down_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.up_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.up_proj.weight,
                    bias=layer.mlp.up_proj.bias,
                    dtype=layer.mlp.up_proj.weight.dtype,
                    device=(
                        layer.mlp.up_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )

            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight,
                bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )
        elif model_type in ["baichuan"]:
            for layer in model.model.layers:
                layer.self_attn.W_pack = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.W_pack.weight.to(device),
                    bias=layer.self_attn.W_pack.bias,
                    dtype=layer.self_attn.W_pack.weight.dtype,
                    device=(
                        layer.self_attn.W_pack.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )
                layer.self_attn.o_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.self_attn.o_proj.weight.to(device),
                    bias=layer.self_attn.o_proj.bias,
                    dtype=layer.self_attn.o_proj.weight.dtype,
                    device=(
                        layer.self_attn.o_proj.weight.device
                        if device is None
                        else device
                    ),
                    empty_init=empty_init,
                )

                layer.mlp.gate_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.gate_proj.weight.to(device),
                    bias=layer.mlp.gate_proj.bias,
                    dtype=layer.mlp.gate_proj.weight.dtype,
                    device=(
                        layer.mlp.gate_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.down_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.down_proj.weight.to(device),
                    bias=layer.mlp.down_proj.bias,
                    dtype=layer.mlp.down_proj.weight.dtype,
                    device=(
                        layer.mlp.down_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )
                layer.mlp.up_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.up_proj.weight.to(device),
                    bias=layer.mlp.up_proj.bias,
                    dtype=layer.mlp.up_proj.weight.dtype,
                    device=(
                        layer.mlp.up_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )

            model.lm_head.weight.data = torch.nn.functional.normalize(
                model.lm_head.weight
            )
            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight.to(device),
                # bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )

        elif model_type in ["qwen"]:
            for layer in model.transformer.h:
                # print(layer.attn.c_attn.weight.device)
                layer.attn.c_attn = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.attn.c_attn.weight.to(device),
                    bias=layer.attn.c_attn.bias,
                    dtype=layer.attn.c_attn.weight.dtype,
                    device=(
                        layer.attn.c_attn.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )
                layer.attn.c_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.attn.c_proj.weight.to(device),
                    bias=layer.attn.c_proj.bias,
                    dtype=layer.attn.c_proj.weight.dtype,
                    device=(
                        layer.attn.c_proj.weight.device if device is None else device
                    ),
                    empty_init=empty_init,
                )

                layer.mlp.w1 = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.w1.weight.to(device),
                    bias=layer.mlp.w1.bias,
                    dtype=layer.mlp.w1.weight.dtype,
                    device=layer.mlp.w1.weight.device if device is None else device,
                    empty_init=empty_init,
                )
                layer.mlp.w2 = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.w2.weight.to(device),
                    bias=layer.mlp.w2.bias,
                    dtype=layer.mlp.w2.weight.dtype,
                    device=layer.mlp.w2.weight.device if device is None else device,
                    empty_init=empty_init,
                )
                layer.mlp.c_proj = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    weight=layer.mlp.c_proj.weight.to(device),
                    bias=layer.mlp.c_proj.bias,
                    dtype=layer.mlp.c_proj.weight.dtype,
                    device=layer.mlp.c_proj.weight.device if device is None else device,
                    empty_init=empty_init,
                )

            model.lm_head = QuantizedLinear(
                weight_bit_width=weight_bit_width * 2,
                weight=model.lm_head.weight.to(device),
                bias=model.lm_head.bias,
                dtype=model.lm_head.weight.dtype,
                device=model.lm_head.weight.device if device is None else device,
                empty_init=empty_init,
            )

        else:
            pass

    return model


def find_linear_layers(module, prefix=""):
    """获取模型中所有的 torch.nn.Linear 和 fairseq2.nn.projection.Linear 层的名字"""
    import fairseq2

    fLinear = fairseq2.nn.projection.Linear
    linear_layers = []
    # 来遍历模型并找到所有的Linear层
    for name, layer in module.named_children():
        # 完整名称
        full_name = prefix + name
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, fLinear):
            linear_layers.append((full_name, layer))
        else:
            # 递归地检查子模块
            linear_layers.extend(find_linear_layers(layer, full_name + "."))
    return linear_layers


def analysis_linear_layer(model):
    """分析模型中所有的 torch.nn.Linear 和 fairseq2.nn.projection.Linear 层"""
    # 获取所有的Linear层
    linear_layers = find_linear_layers(model)

    # 打印所有的Linear层
    for name, layer in linear_layers:
        print(f"Layer name: {name}, Layer: {layer}")


def quantize_all_linear_layer(model, weight_bit_width, empty_init, device):
    """遍历模型中的所有 torch.nn.Linear 和 fairseq2.nn.projection.Linear 层，并替换它们为量化后的版本"""
    global quant_count
    quant_count = 0

    def quantize_linear(module, prefix=""):
        global quant_count
        import fairseq2

        fLinear = fairseq2.nn.projection.Linear

        # 来遍历模型并找到所有的Linear层
        for name, linear_layer in module.named_children():
            # 完整名称
            full_name = prefix + name
            if isinstance(linear_layer, torch.nn.Linear) or isinstance(
                linear_layer, fLinear
            ):
                print(f"Quantizing({quant_count}) layer {full_name}")
                quant_count += 1

                module._modules[name] = QuantizedLinear(
                    weight_bit_width=weight_bit_width,
                    # weight=layer.self_attention.query_key_value.weight.to(torch.cuda.current_device()),
                    weight=linear_layer.weight,
                    bias=linear_layer.bias,
                    dtype=linear_layer.weight.dtype,
                    device=linear_layer.weight.device if device is None else device,
                    empty_init=empty_init,
                )
            else:
                # 递归地检查子模块
                linear_layer = quantize_linear(linear_layer, full_name + ".")
        return module

    model = quantize_linear(model)
    return model


def quantize(
    model,
    weight_bit_width,
    empty_init=False,
    device=None,
    model_type=None,
    lm_hea_quant=False,
):
    """Replace fp16 linear with quantized linear"""

    # msg = "-" * 6 + "quantization Agent3 OfflineWav2VecBertEncoderAgent" + "-" * 6
    # print(msg)
    analysis_linear_layer(model)
    # Quantize the linear layers
    model = quantize_all_linear_layer(model, weight_bit_width, empty_init, device)
    # print(model)
    return model

