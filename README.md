# onnxdiff

Comparison of onnx models by structure, initializers and onnxruntime


## 1 Structs & Parameters

Calculate the match score of the two input onnx models as by parsing the initializers, inputs, outputs, all nodes, and all other fields of the two input onnx models.

* Use the onnx.checker.check_model() interface to check if the input models are reasonable
* Calculate the graph matching score
* node matching score of the input models
* generate a structured diff result

results match:

```bash
Exact Match (100.0%)

╭────────────────────┬─────────┬─────────╮
│ Matching Fields    │ A       │ B       │
├────────────────────┼─────────┼─────────┤
│ Graph.Initializers │ 47/47   │ 47/47   │
│ Graph.Inputs       │ 3/3     │ 3/3     │
│ Graph.Outputs      │ 5/5     │ 5/5     │
│ Graph.Nodes        │ 134/134 │ 134/134 │
│ Graph.Misc         │ 5/5     │ 5/5     │
│ Misc               │ 10/10   │ 10/10   │
╰────────────────────┴─────────┴─────────╯
```


----------


results mismatch:

```bash
Difference Detected (99.915634%)

╭────────────────────┬────────┬────────╮
│ Matching Fields    │ A      │ B      │
├────────────────────┼────────┼────────┤
│ Graph.Initializers │ 17/55  │ 17/59  │
│ Graph.Inputs       │ 0/1    │ 0/4    │
│ Graph.Outputs      │ 0/5    │ 0/5    │
│ Graph.Nodes        │ 77/176 │ 77/199 │
│ Graph.Misc         │ 5/6    │ 5/6    │
│ Misc               │ 10/10  │ 10/10  │
╰────────────────────┴────────┴────────╯
```


## 2 OnnxRuntime


For the given two input onnx models, generate identical random inputs, use onnxruntime to compute the outputs of the onnx models, and compare all the outputs of the two onnx models for consistency.

Results Match:

```bash
OnnxRuntime results:

╭────────────────────────────┬──────────────╮
│ Output Nodes               │   Cosine_Sim │
├────────────────────────────┼──────────────┤
│ Output.Logits              │            1 │
│ Output.Past_key_values     │            1 │
│ Output.Onnx::unsqueeze_84  │            1 │
│ Output.Onnx::unsqueeze_601 │            1 │
│ Output.Onnx::unsqueeze_461 │            1 │
╰────────────────────────────┴──────────────╯
model outputs verify complete:  True
```



----------


Results Mismatch:

```bash
Model output number mismatched

OnnxRuntime results:

╭────────────────────────────┬──────────────────╮
│ Output Nodes               │ Cosine_Sim       │
├────────────────────────────┼──────────────────┤
│ Output.Logits              │ (1, 512, 65024)  │
│ Output.Past_key_values     │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_84  │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_601 │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_461 │ (512, 1, 2, 128) │
╰────────────────────────────┴──────────────────╯

OnnxRuntime results:

╭────────────────────────────┬──────────────────╮
│ Output Nodes               │ Cosine_Sim       │
├────────────────────────────┼──────────────────┤
│ Output.Logits              │ (1, 511, 65024)  │
│ Output.Onnx::unsqueeze_264 │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_265 │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_656 │ (512, 1, 2, 128) │
│ Output.Onnx::unsqueeze_657 │ (512, 1, 2, 128) │
╰────────────────────────────┴──────────────────╯
model outputs verify complete:  False
```


## 3 Install

### 3.1 Install from pip

```bash
python3 -m pip install onnxdiffer
```


### 3.2 Install from source code

```bash
git clone https://github.com/Taot-chen/onnx-diff.git
cd onnx-diff
python3 setup.py sdist bdist_wheel
python3 -m pip install ./dist/*.whl
```




## 4 How To Use

### 4.1 Use in Console

```bash
onnxdiff --onnx_a=/path/to/onnx_a.onnx --onnx_b=/path/to/onnx_b.onnx --ort=1 --detial=1
```

more params:

```bash
onnxdiff --help
usage: onnxdiff [-h] [--onnx_a ONNX_A] [--onnx_b ONNX_B] [--struct STRUCT] [--ort ORT] [--detial DETIAL] [--random_seed RANDOM_SEED]

options:
  -h, --help            show this help message and exit
  --onnx_a ONNX_A       ONNX model a to compare
  --onnx_b ONNX_B       ONNX model b to compare
  --struct STRUCT       compare with structs and parameters
  --ort ORT             compare with onnxruntime
  --detial DETIAL       show detials while mismatch
  --random_seed RANDOM_SEED
                        random seeed for random input
```


### 4.2 Use in python

```python
import onnxdiff
ret = onnxdiff.differ("/path/to/onnx_a.onnx", "/path/to/onnx_b.onnx")
print("ret: ", ret)
```


## Features

- [x] struct & parameters
- [x] onnxruntime
- [ ] not match details
    - [ ] readable details for structrue
- [x] standardize output
- [x] for pypi wheel
    - [x] interface
    - [x] console command
- [ ] Performance Optimization
- [ ] Relay IR compare
- [ ] MLIR compare

