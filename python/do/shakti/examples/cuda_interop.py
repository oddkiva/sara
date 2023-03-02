from pathlib import Path

import numpy as np

import cv2

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray


THIS_DIR = Path(__file__).parent
PROJECT_DIR = (THIS_DIR / ".."  / ".." / ".." / "..").resolve()
IMAGE_PATH = PROJECT_DIR / "data" / "sunflowerField.jpg"
assert IMAGE_PATH.exists()

hwc_image_cpu = cv2.imread(str(IMAGE_PATH))
h, w, c = hwc_image_cpu.shape

hwc_image_gpu = GPUArray((h, w, c), dtype=np.uint8)
hwc_image_gpu.set(hwc_image_cpu)

chw_image_32f_gpu = GPUArray((c, h, w), dtype=np.float32)

chw_image_32f_resized_gpu = GPUArray((c, 512, 960), dtype=np.float32)

cuda_source_filepath = THIS_DIR / "cuda_kernels.cu"
assert cuda_source_filepath.exists()
with open(cuda_source_filepath, 'r') as f:
    cuda_source = f.read()
mod = SourceModule(cuda_source)


func = mod.get_function("from_hwc_uint8_to_chw_float")
func(chw_image_32f_gpu, hwc_image_gpu, np.int32(w), np.int32(h),
     block=(32, 32, 1), grid=(round(w / 32), round(h / 32), 1))

downsample = mod.get_function("naive_downsample")
downsample(chw_image_32f_resized_gpu, chw_image_32f_gpu,
           np.int32(960), np.int32(512),
           np.int32(1600), np.int32(1200),
           block=(32, 32, 1), grid=(round(960 / 32), round(512 / 32), 3))
import ipdb; ipdb.set_trace()

# Inspect visually the CUDA implementation.
hwc_image_32f_cpu = np.transpose(chw_image_32f_resized_gpu.get(), (1, 2, 0))
cv2.imshow("display", hwc_image_32f_cpu)
cv2.waitKey(0)
