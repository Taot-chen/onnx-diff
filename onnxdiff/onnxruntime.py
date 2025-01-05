import onnx
import argparse

from onnx_diff.diff import OnnxDiff
from get_onnx_data import get_node_initializer_name, get_initializer_name_list, get_onnx_data
import os
import onnx.helper as helper
import onnxruntime
from collections import OrderedDict
import  numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_diff(onnx_path0, onnx_path1):
    model0 = onnx.load(onnx_path0)
    model1 = onnx.load(onnx_path1)
    diff = OnnxDiff(model0, model1,)
    results = diff.summary(output=True)
    for i in range(min(len(model0.graph.node), len(model1.graph.node))):
        if model0.graph.node[i].name != model1.graph.node[i].name or model0.graph.node[i].op_type != model1.graph.node[i].op_type:
            print("{} <--> {}".format(model0.graph.node[i], model1.graph.node[i]))
    node_initializer_name0 = get_node_initializer_name(model0)
    initializer_name_list0 = get_initializer_name_list(model0)
    node_initializer_name1 = get_node_initializer_name(model1)
    initializer_name_list1 = get_initializer_name_list(model1)
    for i in range(min(len(initializer_name_list0), len(initializer_name_list1))):
        initializer_name0 = initializer_name_list0[i]
        initializer_name1 = initializer_name_list1[i]
        if '.scales' in initializer_name0:
            continue
        graph_node_name0 = initializer_name0
        graph_node_name1 = initializer_name1
        for key in node_initializer_name0.keys():
            if node_initializer_name0[key] == initializer_name0 and node_initializer_name1[key] == initializer_name1:
                graph_node_name0 = key
                graph_node_name1 = key
        initializer_data0 = get_onnx_data(model0, initializer_name0)
        initializer_data1 = get_onnx_data(model1, initializer_name1)
        if initializer_data0 is None and initializer_name1 is None:
            continue
        if "onnx::MatMul" in initializer_name0:
            initializer_data0 = initializer_data0.transpose(-1, -2)
            initializer_data1 = initializer_data1.transpose(-1, -2)
            if initializer_data0.all() != initializer_data1.all():
                print("graph_node_name0: ", graph_node_name0)
                print("graph_node_name1: ", graph_node_name1)
                print("False")
                print(initializer_data0 == initializer_data1)
            elif graph_node_name0 == graph_node_name1:
                print("graph_node_name: ", graph_node_name0)
                print("weight valid: ", initializer_data0.all() == initializer_data1.all())
    print(results)

def get_onnx_node_out(onnx_file, save_onnx):
    model = onnx.load(onnx_file)
    out_names=[]
    for i, node in enumerate(model.graph.node):
        out_names.append(node.output[0])
    for out_name in out_names:
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = out_name
        model.graph.output.append(intermediate_layer_value_info)
    onnx.save(model, save_onnx, save_as_external_data=True, all_tensors_to_one_file=True)

def get_onnx_matmul_node_out(onnx_file, save_onnx):
    model = onnx.load(onnx_file)
    out_names=[]
    node_types = []
    for i, node in enumerate(model.graph.node):
        out_names.append(node.output[0])
        node_types.append(node.op_type)
    for i in range(len(out_names)):
        out_name = out_names[i]
        if "MatMul" in node_types[i]:
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = out_name
            model.graph.output.append(intermediate_layer_value_info)
    onnx.save(model, save_onnx, save_as_external_data=True, all_tensors_to_one_file=True)

def onnxruntime_infer(onnx_path, output_dir, input_ids_file, attn_mask_file, pos_ids_file, past_key_values_file):
 
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    for i in range(4):
        print(session.get_inputs()[i].name)
    outputs = [x.name for x in session.get_outputs()]
    ort_outs = session.run(outputs, {session.get_inputs()[0].name: input_ids_file, session.get_inputs()[1].name: attn_mask_file, 
                                     session.get_inputs()[2].name: pos_ids_file, session.get_inputs()[3].name: past_key_values_file})
    ort_outs = OrderedDict(zip(outputs, ort_outs))
 
    # For debug
    for key in ort_outs:
        val = ort_outs[key]
        file = output_dir + "/" + key.split("/")[-1] +".npy"
        np.save(file, val, allow_pickle=True, fix_imports=True)

def verify_outputs(output0, output1):
    outputs0 = os.listdir(output0)
    outputs1 = os.listdir(output1)
    output_files0 = []
    output_files1 = []
    for item in outputs0:
        output_files0.append(output0 + "/" + item)
    for item in outputs1:
        output_files1.append(output1 + "/" + item)
    if len(output_files0) != len(output_files1):
        return False
    else:
        for i in range(len(outputs0)):
            for item in output_files1:
                if outputs0[i] in item:
                    data0 = np.load(output_files0[i]).reshape(1, -1)
                    data1 = np.load(item).reshape(1, -1)
                    result = True
                    if data0.shape != data1.shape:
                        result = False
                        return False
                    if cosine_similarity(data0, data1) < 0.99999:
                        for index in range(data0.shape[1]):
                            if np.abs(data0[0][index] - data1[0][index]) > 1e-6:
                                result = False
                    print(f"{outputs0[i].split('.')[0]} cosine_sim: {cosine_similarity(data0, data1)}\t {result}")
    return True
        
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path0", default="", type=str)
    parser.add_argument("--onnx_path1", default="", type=str)

    args = parser.parse_args()
    get_diff(args.onnx_path0, args.onnx_path1)
    
    onnx_with_output0 = "./rank0_2layer_sim_ag_440_with_output/"
    onnx_with_output1 = "./rank0_2layer_sim_ag_445_with_output/"
    new_onnx_with_output0 = "./rank0_2layer_sim_ag_440_with_output_new/"
    new_onnx_with_output1 = "./rank0_2layer_sim_ag_445_with_output_new/"
    output_dir0 = "./onnx_output0/"
    output_dir1 = "./onnx_output1/"
    os.makedirs(onnx_with_output0, exist_ok=True)
    os.makedirs(onnx_with_output1, exist_ok=True)
    os.makedirs(new_onnx_with_output0, exist_ok=True)
    os.makedirs(new_onnx_with_output1, exist_ok=True)
    os.makedirs(output_dir0, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)
    onnx_with_output0 = onnx_with_output0 + "model-iter1_rank0_sim_ag.onnx"
    onnx_with_output1 = onnx_with_output1 + "model-iter1_rank0_sim_ag.onnx"
    new_onnx_with_output0 = new_onnx_with_output0 + "model-iter1_rank0_sim_ag.onnx"
    new_onnx_with_output1 = new_onnx_with_output1 + "model-iter1_rank0_sim_ag.onnx"
    
    input_ids_file = "./dummy_input.npy"
    input_ids_data = np.random.rand(16).astype(np.int64).reshape(16, 1)
    np.save(input_ids_file, input_ids_data)
    attn_mask_file = "./attn_mask.npy"
    attn_mask_data = np.random.rand(16*4096).astype(np.int64).reshape(16, 4096)
    np.save(attn_mask_file, attn_mask_data)
    pos_ids_file = "./pos_ids.npy"
    pos_ids_data = np.random.rand(16).astype(np.int64).reshape(16, 1)
    np.save(pos_ids_file, pos_ids_data)
    past_key_values_file = "./past_key_values.npy"
    past_key_values_data = np.random.rand(2*2*16*1*4095*128).astype(np.float32).reshape(2, 2, 16, 1, 4095, 128)
    np.save(past_key_values_file, past_key_values_data)
    # get_onnx_matmul_node_out(args.onnx_path0, onnx_with_output0)
    # get_onnx_matmul_node_out(args.onnx_path1, onnx_with_output1)
    # allgather2concat(onnx_with_output0, new_onnx_with_output0, 16)
    # allgather2concat(onnx_with_output1, new_onnx_with_output1, 16)
    allgather2concat(args.onnx_path0, new_onnx_with_output0, 16)
    allgather2concat(args.onnx_path1, new_onnx_with_output1, 16)
    onnxruntime_infer(new_onnx_with_output0, output_dir0, input_ids_data, attn_mask_data, pos_ids_data, past_key_values_data)
    onnxruntime_infer(new_onnx_with_output1, output_dir1, input_ids_data, attn_mask_data, pos_ids_data, past_key_values_data)
    verify_result = verify_outputs(output_dir0, output_dir1)
    print("model outputs verify complete: ", verify_result)