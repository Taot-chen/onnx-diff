import argparse
import onnx
from onnxdiff.structs_parameters import OnnxDiff
from onnxdiff.ort_infer import verify_outputs

def differ(
    onnx_a: str,
    onnx_b: str,
    struct: int = 1,
    ort: int = 1,
    detial: int = 1,
    random_seed: int = 0
) -> bool:
    onnxdiffer = OnnxDiff(onnx.load(onnx_a), onnx.load(onnx_b))
    results = onnxdiffer.summary(output=True)
    verify_result = verify_outputs(
            onnx_a,
            onnx_b,
            random_seed = random_seed,
            detial = detial
        )
    print("model outputs verify complete: ", verify_result)
    return verify_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_a", default="./", type=str, help="ONNX model a to compare")
    parser.add_argument("--onnx_b", default="./", type=str, help="ONNX model b to compare")
    parser.add_argument("--struct", default=1, type=int, help="compare with structs and parameters")
    parser.add_argument("--ort", default=0, type=int, help="compare with onnxruntime")
    parser.add_argument("--detial", default=0, type=int, help="show detials while mismatch")
    parser.add_argument("--random_seed", default=0, type=int, help="random seeed for random input")
    args = parser.parse_args()

    assert(args.onnx_a[-5:] == ".onnx" and args.onnx_b[-5:] == ".onnx"), f"onnx_a and onnx_b are both expected path end with \'.onnx\'"
    
    if args.struct:
        differ = OnnxDiff(onnx.load(args.onnx_a), onnx.load(args.onnx_b))
        results = differ.summary(output=True)

    if args.ort:
        verify_result = verify_outputs(
            args.onnx_a,
            args.onnx_b,
            random_seed = args.random_seed,
            detial = args.detial
        )
        print("model outputs verify complete: ", verify_result)

if __name__ == "__main__":
    main()