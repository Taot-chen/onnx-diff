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
│ Graph.Initializers │ 55/55   │ 55/55   │
│ Graph.Inputs       │ 1/1     │ 1/1     │
│ Graph.Outputs      │ 5/5     │ 5/5     │
│ Graph.Nodes        │ 176/176 │ 176/176 │
│ Graph.Misc         │ 6/6     │ 6/6     │
│ Misc               │ 10/10   │ 10/10   │
╰────────────────────┴─────────┴─────────╯
```


----------


results mismatch with struct difference details:

```bash
Struct details:

Node --> input_ids
Node --> onnx::Unsqueeze_461
Node --> onnx::Unsqueeze_601
Node --> onnx::Unsqueeze_84
Node --> past_key_values
Node --> logits
Node --> /output_layer/MatMul
Node --> /transformer/encoder/layers.1/mlp/dense_4h_to_h/MatMul
Node --> /transformer/encoder/layers.1/mlp/Slice_1
Node --> /transformer/encoder/layers.1/mlp/Slice
Node --> /transformer/encoder/layers.1/mlp/dense_h_to_4h/MatMul
Node --> /transformer/encoder/layers.1/self_attention/dense/MatMul
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape_6
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape_5
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape_4
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape_3
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Where
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape_1
Node --> /transformer/encoder/layers.1/self_attention/core_attention/Reshape
Node --> /transformer/encoder/layers.1/self_attention/Unsqueeze_1
Node --> /transformer/encoder/layers.1/self_attention/Unsqueeze
Node --> Concat_441
Node --> Reshape_440
Node --> Concat_432
Node --> Unsqueeze_431
Node --> Unsqueeze_429
Node --> Add_427
Node --> Mul_426
Node --> Mul_424
Node --> Sub_422
Node --> Mul_421
Node --> Gather_419
Node --> Mul_418
Node --> Gather_416
Node --> Reshape_415
Node --> Slice_413
Node --> Slice_408
Node --> Concat_403
Node --> Reshape_402
Node --> Concat_394
Node --> Unsqueeze_393
Node --> Unsqueeze_391
Node --> Add_389
Node --> Mul_388
Node --> Mul_386
Node --> Sub_384
Node --> Mul_383
Node --> Gather_381
Node --> Mul_380
Node --> Gather_378
Node --> Reshape_377
Node --> Slice_375
Node --> Slice_370
Node --> /transformer/encoder/layers.1/self_attention/Reshape_2
Node --> /transformer/encoder/layers.1/self_attention/Split
Node --> /transformer/encoder/layers.1/self_attention/query_key_value/MatMul
Node --> /transformer/encoder/layers.0/mlp/dense_4h_to_h/MatMul
Node --> /transformer/encoder/layers.0/mlp/Slice_1
Node --> /transformer/encoder/layers.0/mlp/Slice
Node --> /transformer/encoder/layers.0/mlp/dense_h_to_4h/MatMul
Node --> /transformer/encoder/layers.0/self_attention/dense/MatMul
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape_6
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape_5
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape_4
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape_3
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Where
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape_1
Node --> /transformer/encoder/layers.0/self_attention/core_attention/Reshape
Node --> /transformer/encoder/layers.0/self_attention/Unsqueeze_1
Node --> /transformer/encoder/layers.0/self_attention/Unsqueeze
Node --> Concat_252
Node --> Reshape_251
Node --> Concat_243
Node --> Unsqueeze_242
Node --> Unsqueeze_240
Node --> Add_238
Node --> Mul_237
Node --> Mul_235
Node --> Sub_233
Node --> Mul_232
Node --> Gather_230
Node --> Mul_229
Node --> Gather_227
Node --> Reshape_226
Node --> Slice_224
Node --> Slice_219
Node --> Concat_214
Node --> Reshape_213
Node --> Concat_205
Node --> Unsqueeze_204
Node --> Unsqueeze_202
Node --> Add_200
Node --> Mul_199
Node --> Mul_197
Node --> Sub_195
Node --> Mul_194
Node --> Gather_192
Node --> Mul_191
Node --> Gather_189
Node --> Reshape_188
Node --> Slice_186
Node --> Slice_181
Node --> /transformer/encoder/layers.0/self_attention/Reshape_2
Node --> /transformer/encoder/layers.0/self_attention/Split
Node --> /transformer/encoder/layers.0/self_attention/query_key_value/MatMul

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
OnnxRuntime verify complete:  True
```



----------


Results Mismatch with difference ditails:

```bash
OnnxRuntime details:

output logits: shapes (292288,) and (227264,) not aligned: 292288 (dim 0) != 227264 (dim 0)
--------------------------------------------------------------------------------
output past_key_values not match --> cosine_sim: 0.752902492472754, data: [-0.7543257   0.3264123   0.11826951 ... -1.968673    1.4913989
 -0.88640976] vs [ 1.          1.          1.         ... -1.968673    1.4913989
 -0.88640976]
--------------------------------------------------------------------------------
output onnx::Unsqueeze_84 not match --> cosine_sim: 0.13509489554262924, data: [-0.00108174 -0.00108715 -0.00046213 ... -0.00161161 -0.00760783
  0.00137464] vs [ 1.          1.          1.         ... -0.00161161 -0.00760783
  0.00137464]
--------------------------------------------------------------------------------
output onnx::Unsqueeze_601 not match --> cosine_sim: 0.47637683795808006, data: [-1.7323476  -3.2871733  -0.24971327 ...  0.7797358  -0.68161255
  2.057881  ] vs [ 1.          1.          1.         ...  0.7775023  -0.68093836
  2.0550873 ]
--------------------------------------------------------------------------------
output onnx::Unsqueeze_461 not match --> cosine_sim: 0.9704498155269378, data: [-0.18699308  0.3270467  -0.10742211 ... -0.10703173  0.06767441
  0.06302631] vs [ 1.          1.          1.         ... -0.1068389   0.06742625
  0.06372751]
--------------------------------------------------------------------------------

OnnxRuntime results:

╭────────────────────────────┬──────────────╮
│ Output Nodes               │ Cosine_Sim   │
├────────────────────────────┼──────────────┤
│ Output.Logits              │ -            │
│ Output.Past_key_values     │ 0.752902     │
│ Output.Onnx::unsqueeze_84  │ 0.135095     │
│ Output.Onnx::unsqueeze_601 │ 0.476377     │
│ Output.Onnx::unsqueeze_461 │ 0.97045      │
╰────────────────────────────┴──────────────╯
OnnxRuntime verify complete:  False
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
  --detail DETIAL       show details while mismatch
  --input_mode INPUT_MODE   input mode, should be 'random', 'zeros' or 'ones'
  --random_seed RANDOM_SEED
                        random seeed for random input
  --max_diff MAX_DIFF   the maximum cosine similarity diff allowed
```


### 4.2 Use in python

```python
import onnxdiff
ret = onnxdiff.differ(
    "/path/to/onnx_a.onnx",
    "/path/to/onnx_b.onnx",
    struct = 1,
    ort = 1,
    detail = 0,
    input_mode = "random",
    random_seed = 0,
    max_diff = 1e-5
)
```


## Features

- [x] struct & parameters
- [x] onnxruntime
- [x] not match details
    - [x] readable details for structrue
- [x] standardize output
- [x] for pypi wheel
    - [x] interface
    - [x] console command
- [ ] Performance Optimization
- [ ] Relay IR compare
- [ ] MLIR compare
