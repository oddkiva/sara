import tensorrt as trt


logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open("yolox-tiny.bin", "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
