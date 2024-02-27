import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/home/ayush/fyp/ICCV2019_MirrorNet/model_quant.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)