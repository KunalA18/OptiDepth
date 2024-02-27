import numpy as np
import os
import time
import cv2
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet

args = {
    'snapshot': '160',
    'scale': 384,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'MSD': msd_testing_root}

to_pil = transforms.ToPILImage()

def mirror(frame):
    net = MirrorNet()

    net.load_state_dict(torch.load("/home/ayush/fyp/ICCV2019_MirrorNet/ckpt/MirrorNet/160.pth"))

    net.eval()

    with torch.no_grad():
        img = Image.fromarray(frame)
        w, h = img.size
        img_var = Variable(img_transform(img).unsqueeze(0))
        f_4, f_3, f_2, f_1 = net(img_var)
        f_4 = f_4.data.squeeze(0).cpu()
        f_3 = f_3.data.squeeze(0).cpu()
        f_2 = f_2.data.squeeze(0).cpu()
        f_1 = f_1.data.squeeze(0).cpu()
        f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
        f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
        f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
        f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
        if args['crf']:
            f_4 = crf_refine(np.array(img.convert('RGB')), f_4)

        return f_4
    
import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
start_time = time.time()
counter = 0
fps = 0

while True:
    check, frame1 = cam.read()

    frame = mirror(frame1)
    # Invert the Mask
    inv_frame = cv2.bitwise_not(frame)

    cv2.imshow('video', inv_frame)

    counter += 1
    if (time.time() - start_time) > 1:
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()
        print(fps)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()