from pathlib import Path
import tensorrt as trt


ONNX_MODEL_DIR_PATH = Path("/home/david/Downloads/ha-onnx/")
ONNX_MODEL_PATH = ONNX_MODEL_DIR_PATH / "output/export_results/yolox-tiny.onnx"
assert ONNX_MODEL_PATH.exists()

logger = trt.Logger(trt.Logger.WARNING)

network_builder = trt.Builder(logger)
network = network_builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

onnx_parser = trt.OnnxParser(network, logger)
parsed_successfully = onnx_parser.parse_from_file(str(ONNX_MODEL_PATH))
for idx in range(onnx_parser.num_errors):
    print("ERROR {}: {}".format(idx, onnx_parser.get_error(idx)))
if not parsed_successfully:
    exit(1)

trt_config = network_builder.create_builder_config()
# trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (1 << 30))

trt_profile = network_builder.create_optimization_profile()
trt_profile.set_shape("input",
                      (1, 3, 416, 416),
                      (1, 3, 512, 960),
                      (1, 3, 1024, 1920))
trt_config.add_optimization_profile(trt_profile)


serialized_engine = network_builder.build_serialized_network(network,
                                                             trt_config)
serialized_engine_filepath = "{}.bin".format(ONNX_MODEL_PATH.stem)
with open(serialized_engine_filepath, "wb") as f:
    f.write(serialized_engine)
