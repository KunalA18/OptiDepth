import cv2
import numpy as np
import time
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
from misc import check_mkdir, crf_refine
import matplotlib.pyplot as plt
from torch.autograd import Variable

to_pil = transforms.ToPILImage()

def to_numpy(tensor):
    if tensor.requires_grad:
        print("hello")
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Start Session
ort_session = onnxruntime.InferenceSession("/home/ayush/fyp/model.onnx")
# Model Info
input_name = ort_session.get_outputs()[0].shape
print('Input Name:', input_name)

img = Image.open("/home/ayush/fyp/CVPR2020_GDNet/image/glass.jpeg")
resize = transforms.Resize([416, 416])
w, h = img.size
# img_ = resize(img)
# to_tensor = transforms.ToTensor()
img_y = Variable(img_transform(img).unsqueeze(0), requires_grad=False).to("cuda")
print(to_numpy(img_y))
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0].squeeze(0)
print(img_out_y)
f1 = np.array(transforms.Resize((h, w))(to_pil(img_out_y[0])))
pred_mask = (f1*255).astype(np.uint8)
f3 = crf_refine(np.array(img), pred_mask)
# grayImage = cv2.cvtColor(img_out_y[0])
plt.imshow(Image.fromarray(f3))
plt.show()

# img_transform = transforms.Compose([
#     transforms.Resize((384,384)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def to_numpy(tensor):
#     if tensor.requires_grad:
#         print("hello")
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # Start Session
# ort_session = onnxruntime.InferenceSession("model_m.onnx", providers=["CPUExecutionProvider"])
# # Model Info
# input_name = ort_session.get_outputs()[0].shape
# print('Input Name:', input_name)

# # img = Image.open("/home/ayush/fyp/ICCV2019_MirrorNet/MSD/test/image/5537_512x640.jpg")
# # resize = transforms.Resize([384, 384])
# # img = resize(img)
# # to_tensor = transforms.ToTensor()
# # img_y = to_tensor(img)
# # img_y.unsqueeze_(0)
# # print(to_numpy(img_y))
# # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
# # ort_outs = ort_session.run(None, ort_inputs)
# # img_out_y = ort_outs[0].squeeze(0)
# # print(img_out_y)
# # grayImage = cv2.cvtColor(img_out_y[0], cv2.COLOR_GRAY2BGR)
# # plt.imshow(grayImage)
# # plt.show()

# cam = cv2.VideoCapture(0)
# start_time = time.time()
# counter = 0
# fps = 0

# while True:
#     check, frame1 = cam.read()
#     img = Image.fromarray(frame1)

#     # cv2.imshow('video', img_transform(img).numpy().transpose(1, 2, 0)) #Display Tensor
    
#     img_tensor = img_transform(img).unsqueeze(0)
#     ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
#     output_mask_list = ort_session.run(None, ort_inputs)
#     f1 = output_mask_list[0].squeeze(0)

#     cv2.imshow('video', f1[0])

#     counter += 1
#     if (time.time() - start_time) > 1:
#         fps = counter / (time.time() - start_time)

#         counter = 0
#         start_time = time.time()
#         print(fps)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cam.release()
# cv2.destroyAllWindows()