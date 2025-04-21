import argparse
import onnx
from structs_parameters import OnnxDiff
from ort_infer import verify_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_a", default="./", type=str, help="ONNX model a to compare")
    parser.add_argument("--onnx_b", default="./", type=str, help="ONNX model b to compare")
    parser.add_argument("--struct", default=1, type=int, help="compare with structs and parameters")
    parser.add_argument("--ort", default=0, type=int, help="compare with onnxruntime")
    args = parser.parse_args()

    assert(args.onnx_a[-5:] == ".onnx" and args.onnx_b[-5:] == ".onnx"), f"onnx_a and onnx_b are both expected path end with \'.onnx\'"
    
    if args.ort:
        verify_result = verify_outputs(args.onnx_a, args.onnx_b)
        print("model outputs verify complete: ", verify_result)
    if args.struct:
        differ = OnnxDiff(onnx.load(args.onnx_a), onnx.load(args.onnx_b))
        results = differ.summary(output=True)
