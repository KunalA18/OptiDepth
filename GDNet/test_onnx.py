# import onnxruntime
# import numpy as np

# from torch import nn
# import torch.utils.model_zoo as model_zoo
# import torch.onnx
# import torchvision.transforms as transforms
# import onnx

# import cv2
# import numpy as np
# import time
# import onnxruntime
# import torchvision.transforms as transforms
# from PIL import Image
# from misc import check_mkdir, crf_refine
# import matplotlib.pyplot as plt
# from gdnet import GDNet

# torch_model = GDNet() 
# path = "/home/kunal/Downloads/FYP/OptiDepth/GDNet/200.pth" 
# torch_model.load_state_dict(torch.load(path)) 

# # set the model to inference mode 
# torch_model.eval() 

# x = torch.randn(1, 3, 224, 224, requires_grad=True)

# torch_out = torch_model(x)

# ort_session = onnxruntime.InferenceSession('gdnet.onnx', providers=["CPUExecutionProvider"])

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")


# ----------------------------

import cv2
import numpy as np
import time
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
from misc import check_mkdir, crf_refine
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if tensor.requires_grad:
        print("hello")
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Start Session
ort_session = onnxruntime.InferenceSession('gdnet.onnx', providers=["CPUExecutionProvider"])
# Model Info
input_name = ort_session.get_outputs()[0].shape
print('Input Name:', input_name)

img = Image.open("/home/kunal/Downloads/FYP/CVPR2020_GDNet/GDD/test/image/glass.jpeg")

resize = transforms.Resize([224, 224])
img = resize(img)
to_tensor = transforms.ToTensor()
img_y = to_tensor(img)
img_y.unsqueeze_(0)
# print(to_numpy(img_y))
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0].squeeze(0)
# print(img_out_y)
grayImage = cv2.cvtColor(img_out_y[0], cv2.COLOR_GRAY2BGR)
plt.imshow(grayImage)
plt.show()

