import argparse
import onnx
from structs_parameters import OnnxDiff

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_a", default="./", type=str, help="ONNX model a to compare")
    parser.add_argument("--onnx_b", default="./", type=str, help="ONNX model b to compare")
    args = parser.parse_args()

    assert(args.onnx_a[-5:] == ".onnx" and args.onnx_b[-5:] == ".onnx"), f"onnx_a and onnx_b are both expected path end with \'.onnx\'"
    
    model0 = onnx.load(args.onnx_a)
    model1 = onnx.load(args.onnx_b)
    differ = OnnxDiff(model0, model1)
    results = differ.summary(output=True)

if __name__ == "__main__":
    main()

