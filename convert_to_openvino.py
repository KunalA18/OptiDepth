import openvino as ov
import onnx

onnx_model = onnx.load("model_gdnet.onnx")
onnx.checker.check_model(onnx_model, True)

ov_model = ov.convert_model("/home/ayush/OptiDepth/model_gdnet.onnx")
ov.save_model(ov_model, "model_gdnet.xml", compress_to_fp16=True)
print("Exported.")

import blobconverter

NN_PATH = blobconverter.from_openvino(
    "/home/ayush/OptiDepth/model_gdnet.xml", "/home/ayush/OptiDepth/model_gdnet.bin", shaves=6)

print(NN_PATH)
    
# "stderr": "Cannot create Interpolate layer /h5_up/Resize id:969 from unsupported opset: opset11\n",
# "stdout": "OpenVINO Runtime version ......... 2022.1.0\nBuild ........... 2022.1.0-7019-cdb9bec7210-releases/2022/1\n"