import cv2
import depthai as dai
import numpy as np
import argparse
import time
from misc import crf_refine

# --------------- Pipeline ---------------
NN_WIDTH, NN_HEIGHT = 224, 224
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Define a neural network
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath("/home/kunal/Downloads/FYP/OptiDepth/GDNet/gdnet_openvino_2021.4_5shave.blob")
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# Link
cam.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_cam.input)
detection_nn.out.link(xout_nn.input)


# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_frame = q_cam.get()

        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get output layer
        pred_mask = np.array(in_nn.getFirstLayerFp16()).reshape((NN_HEIGHT, NN_WIDTH))
        pred_mask = (pred_mask*255).astype(np.uint8)

        f_1 = crf_refine(np.array(frame), pred_mask)

        cv2.imshow("Mask", f_1)

        # Calculate FPS
        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()
            print(fps)

        if cv2.waitKey(1) == ord('q'):
            break