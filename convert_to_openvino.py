import openvino as ov

ov_model = ov.convert_model("/home/ayush/fyp/model.onnx")
ov.save_model(ov_model, "model.xml", compress_to_fp16=True)
print("Exported.")

# import blobconverter

# NN_PATH = blobconverter.from_openvino(
#     "/home/ayush/fyp/model.xml", "/home/ayush/fyp/model.bin", shaves=6)

# print(NN_PATH)
    
# "stderr": "Cannot create Interpolate layer /h5_up/Resize id:969 from unsupported opset: opset11\n",
# "stdout": "OpenVINO Runtime version ......... 2022.1.0\nBuild ........... 2022.1.0-7019-cdb9bec7210-releases/2022/1\n"