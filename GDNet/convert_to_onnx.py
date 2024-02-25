import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from gdnet import GDNet
# import onnx

torch_model = GDNet() 
path = "/home/kunal/Downloads/FYP/OptiDepth/GDNet/200.pth" 
torch_model.load_state_dict(torch.load(path)) 

# set the model to inference mode 
torch_model.eval() 

x = torch.randn(1, 3, 416, 416, requires_grad=True)
torch_out = torch_model(x)

# onnx_model = onnx.load("model.onnx")
# onnx.checker.check_model(onnx_model, True)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "gdnetconv.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print('Model has been converted to ONNX') 





    
