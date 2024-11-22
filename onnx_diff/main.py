import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_0", default="./", type=str, help="ONNX model 0 to compare")
    parser.add_argument("--onnx_1", default="./", type=str, help="ONNX model 1 to compare")
    args = parser.parse_args()

    assert(args.onnx_0[-5:] == ".onnx" and args.onnx_1[-5:] == ".onnx"), f"onnx_0 and onnx_1 are both expected path end with \'.onnx\'"

if __name__ == "__main__":
    main()

