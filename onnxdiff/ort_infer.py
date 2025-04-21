import argparse
from utils import memory_efficient_cosine

import onnxruntime as ort
import  numpy as np
from typing import Dict

def onnx_type_to_numpy(onnx_type: str) -> np.dtype:
    type_map = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return type_map.get(onnx_type, np.float32)


def handle_dynamic_shape(
    shape: list,
    dynamic_override: Dict[str, int] = None
) -> list:
    dynamic_values = {
        "batch_size": 1,
        "seq_len": 1
    }
    if dynamic_override:
        dynamic_values.update(dynamic_override)
    
    handled = []
    for dim in shape:
        if isinstance(dim, int):
            handled.append(dim)
        elif isinstance(dim, str) and dim in dynamic_values:
            handled.append(dynamic_values[dim])
        else:
            handled.append(1)
    return handled


def generate_input_data(
    shape: list,
    dtype: np.dtype,
    mode: str = "random",
    random_seed: int = 0
) -> np.ndarray:
    if mode == "random":
        np.random.seed(random_seed)
        if np.issubdtype(dtype, np.floating):
            return np.random.randn(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 100, size=shape, dtype=dtype)
        elif dtype == np.bool_:
            return np.random.choice([True, False], size=shape)
    elif mode == "zeros":
        return np.zeros(shape, dtype=dtype)
    elif mode == "ones":
        return np.ones(shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def onnxruntime_infer(
    onnx_path: str,
    input_mode: str = "random",
    dynamic_override: Dict[str, int] = None,
    random_seed: int = 0
) -> Dict[str, np.ndarray]:

    """
    执行 ONNX 模型推理
    参数:
        model_path: ONNX 模型路径
        input_mode: 输入生成模式 ("random"|"zeros"|"ones")
        dynamic_override: 动态维度覆盖值，如 {"batch_size": 4}
    返回:
        {输出名称: 输出值} 的字典
    """

    options = ort.SessionOptions()
    options.log_severity_level = 3  # 3=ERROR, 2=WARNING（默认）, 1=INFO, 0=VERBOSE
    session = ort.InferenceSession(onnx_path, sess_options=options)
    input_dict = {}
    for input_info in session.get_inputs():
        shape = handle_dynamic_shape(input_info.shape, dynamic_override)
        dtype = onnx_type_to_numpy(input_info.type)
        input_data = generate_input_data(shape, dtype, mode = input_mode, random_seed = random_seed)
        input_dict[input_info.name] = input_data

    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, input_dict)
    return {name: value for name, value in zip(output_names, outputs)}

def verify_outputs(
    onnx_a: str,
    onnx_b: str,
    random_seed: int = 0
) -> bool:
    outputs_a = onnxruntime_infer(onnx_a)
    outputs_b = onnxruntime_infer(onnx_b)
    if outputs_a.keys() != outputs_b.keys():
        return False
    else:
        for output_name in outputs_a.keys():
            cosine_sim = memory_efficient_cosine(outputs_a[output_name].flatten(), outputs_b[output_name].flatten())
            if cosine_sim < 0.99:
                print(f"output {output_name} not match --> cosine_sim: {cosine_sim}")
                return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_a", default="", type=str)
    parser.add_argument("--onnx_b", default="", type=str)

    args = parser.parse_args()
    verify_result = verify_outputs(args.onnx_a, args.onnx_b)
    print("model outputs verify complete: ", verify_result)
