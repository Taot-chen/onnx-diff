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



## Features

- [x] struct & parameters
- [x] onnxruntime
- [x] not match details
- [x] standardize output
- [ ] for pypi wheel
    - [ ] interface
    - [ ] console command
