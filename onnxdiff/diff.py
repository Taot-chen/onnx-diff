import argparse
import random

import onnx
from onnxdiff.structs_parameters import OnnxDiff
from onnxdiff.ort_infer import verify_outputs

# here, all params are expected to have a default value, and can be used by py package
def differ(onnx_a: str,
    onnx_b: str,
    struct: int = 1,
    ort: int = 1,
    detial: int = 1,
    random_seed: int = 0
) -> bool:
    onnxdiffer = OnnxDiff(onnx.load(onnx_a), onnx.load(onnx_b))
    onnxdiffer.summary(output=True)
    verify_result = verify_outputs(
            onnx_a,
            onnx_b,
            random_seed = random_seed,
            detail = detial,
            input_mode = "random",
            max_diff = 1e-6
        )
    print("model outputs verify complete: ", verify_result)
    return verify_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_a", default="./", type=str, help="ONNX model a to compare")
    parser.add_argument("--onnx_b", default="./", type=str, help="ONNX model b to compare")
    parser.add_argument("--struct", default=1, type=int, help="compare with structs and parameters")
    parser.add_argument("--ort", default=0, type=int, help="compare with onnxruntime")
    parser.add_argument("--detail", default=0, type=int, help="show details while mismatch")
    parser.add_argument("--input_mode", default="random", type=str, help="input mode, should be 'random', 'zeros' or 'ones'")
    parser.add_argument("--random_seed", default=random.randint(0, 2**10), type=int, help="random seed for random input")
    parser.add_argument("--max_diff", default=1e-6, type=float, help="the maximum cosine similarity diff allowed")
    args = parser.parse_args()

    assert(args.onnx_a[-5:] == ".onnx" and args.onnx_b[-5:] == ".onnx"), f"onnx_a and onnx_b are both expected path end with \'.onnx\'"
    
    if args.struct:
        onnx_differ = OnnxDiff(onnx.load(args.onnx_a), onnx.load(args.onnx_b))
        onnx_differ.summary(output=True)
        print("=" * 80)

    if args.ort:
        verify_result = verify_outputs(
            args.onnx_a,
            args.onnx_b,
            random_seed = args.random_seed,
            detail= args.detail,
            input_mode=args.input_mode,
            max_diff=args.max_diff,
        )
        print("model outputs verify complete: ", verify_result)

if __name__ == "__main__":
    main()
