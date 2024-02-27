import cv2
import numpy as np
import time
import onnx
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from mirrornet import MirrorNet

model = MirrorNet()
model.load_state_dict(torch.load("MirrorNet/ckpt/MirrorNet/160.pth"))
model.eval()

dummy_input = Variable(torch.randn(1, 3, 384, 384, requires_grad=True))

with torch.no_grad():
    #save to onnx
    torch.onnx.export(  model,               # model being run
                        dummy_input,                         # model input (or a tuple for multiple inputs)
                        "model_mirrornet.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to 16-Mirrornet
                        do_constant_folding=True ) # whether to execute constant folding for optimization
    #                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                                 'output' : {0 : 'batch_size'}})

    # Run torch model
    torch_out=model(dummy_input)

# onnx_model = onnx.load("model_g.onnx")
# onnx.checker.check_model(onnx_model, True)
ort_session = onnxruntime.InferenceSession("/home/ayush/OptiDepth/model_mirrornet.onnx") #providers=["CPUExecutionProvider"]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
