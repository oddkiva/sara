from pathlib import Path
from typing import List, Tuple

import numpy as np

import seaborn as sns

import cv2

# Import PyCUDA before TensorRT to avoid a runtime error.
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray

import tensorrt as trt

from ensemble_boxes import weighted_boxes_fusion


YOLOX_TINY_DIR_PATH = Path(__file__).parent
YOLOX_TINY_PLAN_PATH = YOLOX_TINY_DIR_PATH / "yolox-tiny.bin"
COLOR_PALETTE = [(round(r*255), round(g * 255), round(b*255))
                 for (r, g, b) in sns.color_palette("Spectral", 7)]

VIDEO_FILEPATH = "/home/david/Desktop/Datasets/sample-1.mp4"


class ImagePreprocessor:

    def __init__(self, input_hwc_shape):
        h, w, c = input_hwc_shape
        self._input_hwc_shape = (h, w, c)
        self._hwc_image_gpu = GPUArray(self._input_hwc_shape, dtype=np.uint8)

        self._ht = 512
        self._wt = 960
        self._chw_image_32f_gpu = GPUArray((c, h, w), dtype=np.float32)
        self._chw_image_32f_resized_gpu = GPUArray((c, self._ht, self._wt), dtype=np.float32)

        cuda_source_filepath = YOLOX_TINY_DIR_PATH / "cuda_kernels.cu"
        assert cuda_source_filepath.exists()
        with open(cuda_source_filepath, 'r') as f:
            cuda_source = f.read()
        mod = SourceModule(cuda_source)


        self._to_chw_float_kernel = mod.get_function("from_hwc_uint8_to_chw_float")
        self._grid = (round(w / 32), round(h / 32), 1)

        self._grid_ds = (round(self._wt / 32), round(self._ht / 32), 3)
        self._downsample_kernel = mod.get_function("naive_downsample")

    def preprocess(self, hwc_image_cpu: np.ndarray, stream: cuda.Stream):
        # Copy to the GPU array.
        self._hwc_image_gpu.set(hwc_image_cpu)

        # Convert the GPU aray To CHW float32 format.
        h, w, _ = self._input_hwc_shape
        self._to_chw_float_kernel(self._chw_image_32f_gpu, self._hwc_image_gpu,
                                  np.int32(w), np.int32(h),
                                  block=(32, 32, 1), grid=self._grid,
                                  stream=stream)

        # Resize the GPU array.
        self._downsample_kernel(self._chw_image_32f_resized_gpu,
                                self._chw_image_32f_gpu,
                                np.int32(self._wt), np.int32(self._ht),
                                np.int32(w), np.int32(h),
                                block=(32, 32, 1), grid=self._grid_ds,
                                stream=stream)

    def get(self, stream: cuda.Stream) -> np.ndarray:
        stream.synchronize()
        hwc_image_32f_cpu = np.transpose(self._chw_image_32f_resized_gpu.get(),
                                         (1, 2, 0)) / 255.
        return hwc_image_32f_cpu


class TrtInferenceEngine:

    def __init__(self, plan_filepath: Path, conf_thresh=0.4):
        self._confidence_thresh = conf_thresh
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(str(plan_filepath), "rb") as f:
            serialized_engine = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
        self._context = self._engine.create_execution_context()

        self._boxes_tensor = GPUArray((1, 10080, 4), np.float32)
        self._classes_tensor = GPUArray((1, 10080, 2), np.float32)
        self._context.set_tensor_address("boxes", self._boxes_tensor.ptr)
        self._context.set_tensor_address("classes", self._classes_tensor.ptr)

    def bind_input(self, name, ptr):
        self._context.set_tensor_address(name, ptr)

    def run(self, stream: cuda.Stream):
        self._context.execute_async_v3(stream.handle)

    def get(self, stream: cuda.Stream) -> Tuple[List, List, List]:
        # Transfer the output tensor data from the GPU memory to the CPU
        # memory.
        stream.synchronize()
        boxes = self._boxes_tensor.get()
        classes = self._classes_tensor.get()

        # Decode the data.
        boxes, confidences, ids = self.decode(boxes, classes)

        # NMS
        boxes, confidences, ids = weighted_boxes_fusion([boxes],
                                                        [confidences],
                                                        [ids])

        return boxes, confidences, ids

    def decode(self,
               boxes_tensor: np.ndarray,
               classes_tensor: np.ndarray) -> Tuple[List[np.ndarray],
                                                    List[np.ndarray],
                                                    List[np.ndarray]]:
        # Make sense of the classes tensor.
        confidences = classes_tensor[..., 0]
        ids = classes_tensor[..., 1]

        # Mask low-confidence detections with the threshold.
        mask = confidences > self._confidence_thresh

        # Filter detections.
        confidences_filtered = confidences[mask]
        ids_filtered = ids[mask]
        boxes_filtered = boxes_tensor[mask]

        return (boxes_filtered.tolist(), confidences_filtered.tolist(),
                ids_filtered.tolist())


def draw_text(frame_annotated: np.ndarray, text: str, origin, font,
              font_scale=0.5, thickness=1):
    cv2.putText(frame_annotated, text, origin, font,
                font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame_annotated, text, origin, font,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


stream = cuda.Stream()

video_stream = cv2.VideoCapture(VIDEO_FILEPATH)
w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_preprocessor = ImagePreprocessor((h, w, 3))
inference_engine = TrtInferenceEngine(YOLOX_TINY_PLAN_PATH)
inference_engine.bind_input("input",
                            image_preprocessor._chw_image_32f_resized_gpu.ptr)

while True:
    read_frame, frame = video_stream.read()
    if not read_frame:
        break

    image_preprocessor.preprocess(frame, stream)

    inference_engine.run(stream)
    boxes, confidences, ids = inference_engine.get(stream)

    # Rescale the boxes to the original input frame
    boxes[:, (0, 2)] *= float(w)
    boxes[:, (1, 3)] *= float(h)

    frame_annotated = np.copy(frame)
    for (box, confidence, id) in zip(boxes, confidences, ids):
        x1, y1, x2, y2 = np.round(box).astype(int)
        id_int = int(id)
        color =  COLOR_PALETTE[id_int]

        # Draw the detection box.
        cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

        # Draw the text.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text = "Class {}  {}%".format(id_int, int(round(confidence * 100)))
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale,
                                                       thickness + 1)
        cv2.rectangle(frame_annotated,
                      (x1, y1 - text_height), (x1 + text_width, y1),
                      color, cv2.FILLED)
        draw_text(frame_annotated, text, (x1, y1), font, font_scale)


    cv2.imshow("video", frame_annotated)
    key = cv2.waitKey(1)
    if key == 27:
        break
