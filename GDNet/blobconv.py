import blobconverter

blob_path = blobconverter.from_onnx(
    model="/home/kunal/Downloads/FYP/OptiDepth/GDNet/model_g.onnx",
    data_type="FP16",
    shaves=5,
)