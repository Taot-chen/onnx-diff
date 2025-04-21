# onnxdiff

Comparison of onnx models by structure, initializers and onnxruntime


## 1 Structs & Parameters

Calculate the match score of the two input onnx models as by parsing the initializers, inputs, outputs, all nodes, and all other fields of the two input onnx models.

* Use the onnx.checker.check_model() interface to check if the input models are reasonable
* Calculate the graph matching score
* node matching score of the input models
* generate a structured diff result

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


## 2 OnnxRuntime


For the given two input onnx models, generate identical random inputs, use onnxruntime to compute the outputs of the onnx models, and compare all the outputs of the two onnx models for consistency.

model outputs verify complete:  True.



## Features

- [x] struct & parameters
- [x] onnxruntime
- [ ] not match details
- [ ] standardize output
- [ ] for pypi wheel
    - [ ] interface
    - [ ] console command
