import sys

from PySide2.QtCore import Qt

import numpy as np

import do.sara as sara
import do.shakti as shakti

import cv2


def user_main():
    video_file = sys.argv[1]
    video_stream = sara.VideoStream()
    video_stream.open(video_file)

    h, w, _ = video_stream.sizes()

    video_frame = np.empty(video_stream.sizes(), dtype=np.uint8)
    video_frame_gray8 = np.empty(video_stream.sizes()[:2], dtype=np.uint8)
    video_frame_gray32f = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    video_frame_convolved = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    gradx = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    grady = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    mag = np.empty(video_stream.sizes()[:2], dtype=np.float32)
    ori = np.empty(video_stream.sizes()[:2], dtype=np.float32)

    ed = sara.EdgeDetector()

    sigma = 3.2
    gauss_trunc = 4
    ksize = int(2 * sigma * gauss_trunc + 1)
    if ksize % 2 == 0:
        ksize += 1

    try:
        shakti_cuda_gaussian_filter = shakti.CudaGaussianFilter(sigma, gauss_trunc)
    except:
        shakti_cuda_gaussian_filter = None

    # ksize has to be <= 32.
    try:
        video_frame_gray32f_cuda = cv2.cuda_GpuMat()
        video_frame_convolved_cuda = cv2.cuda_GpuMat()
        opencv_cuda_gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_32F, cv2.CV_32F, (ksize, ksize), sigma)
    except:
        opencv_cuda_gaussian_filter = None

    # Benchmarking on 4K video against Vanilla OpenCV.
    # Should compare with OpenCV CUDA implementation.
    sara.create_window(w, h)

    while video_stream.read(video_frame):
        # On the normal desktop: Shakti wins on.
        # On the MacBook Air 2018: on par with OpenCV a bit more on the edge.
        with sara.Timer("[SHAKTI] rgb8 to gray32f"):
            shakti.convert_rgb8_to_gray32f_cpu(video_frame, video_frame_gray32f)
        with sara.Timer("[OPENCV] rgb8 to gray32f"):
            cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY, video_frame_gray8)
            video_frame_gray32f = video_frame_gray8.astype(np.float32) / 255
        print()

        # On a nVidia Titan Xp Pascal, in normalized times:
        # 1. Shakti CUDA impl      ~0.18
        # 2. Shakti Halide GPU v2  ~0.25
        # 3. OpenCV GPU            ~0.27
        # 4. Shakti Halide GPU v1  ~0.33
        # 5. Shakti Halide CPU v2  ~0.55
        # 6. Shakti Halide CPU v1  ~0.70
        # 7. OpenCV CPU             1.0
        #
        # Corresponding speed up :
        # 1. Shakti CUDA impl      ~5.56x
        # 2. Shakti Halide GPU v2  ~4.00x
        # 3. OpenCV GPU            ~3.70x
        # 4. Shakti Halide GPU v1  ~3.03x
        # 5. Shakti Halide CPU v2  ~1.82x
        # 6. Shakti Halide CPU v1  ~1.42x
        # 7. OpenCV CPU             1.0
        #
        # Representative timing data:
        # [[SHAKTI][CUDA] gaussian convolution]          Elapsed: 9.571552276611328 ms
        # [[SHAKTI][Halide-GPU] gaussian convolution v2] Elapsed: 13.216495513916016 ms
        # [[OPENCV][CUDA] gaussian convolution]          Elapsed: 14.25933837890625 ms
        # [[SHAKTI][Halide-GPU] gaussian convolution v1] Elapsed: 17.3337459564209 ms
        # [[SHAKTI][Halide-CPU] gaussian convolution v2] Elapsed: 28.859853744506836 ms
        # [[SHAKTI][Halide-CPU] gaussian convolution v1] Elapsed: 37.05859184265137 ms
        # [[OPENCV][CPU] gaussian convolution] Elapsed: 52.51121520996094 ms
        #
        # Shakti CPU v2 and Shakti CUDA are the best implementations.
        # Shakti GPU v2 is 1.08x faster than OpenCV GPU implementation
        #
        # Fundamentally, OpenCV CUDA implementation is very far off from Shakti
        # CUDA which:
        # - is already 1.5x faster and,
        # - has a quite simple implementation.
        # More problematic, OpenCV GPU implementation only supports kernel size <
        # 32, which really limits its usability.
        #
        # So on nVidia platforms, we should prefer coding in CUDA to better
        # leverage all the hardware acceleration.
        #
        # On Macbook Air: OpenCV wins against any Shakti/Halide implementation
        # The CPU-GPU transfer may be the bottleneck.
        with sara.Timer("[SHAKTI][Halide-CPU] gaussian convolution"):
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved, sigma,
                                        gauss_trunc, False)
        with sara.Timer("[SHAKTI][Halide-GPU] gaussian convolution"):
            shakti.gaussian_convolution(video_frame_gray32f,
                                        video_frame_convolved, sigma,
                                        gauss_trunc, True)
        if shakti_cuda_gaussian_filter is not None:
            with sara.Timer("[SHAKTI][CUDA] gaussian convolution"):
                shakti_cuda_gaussian_filter.apply(video_frame_gray32f,
                                                  video_frame_convolved);
        with sara.Timer("[OPENCV][CPU] gaussian convolution"):
            cv2.GaussianBlur(video_frame_gray32f, (ksize, ksize), sigma,
                             video_frame_convolved)
        if opencv_cuda_gaussian_filter is not None:
            with sara.Timer("[OPENCV][CUDA] gaussian convolution"):
                video_frame_gray32f_cuda.upload(video_frame_gray32f)
                opencv_cuda_gaussian_filter.apply(video_frame_gray32f_cuda,
                                                  video_frame_convolved_cuda)
                video_frame_convolved_cuda.download(video_frame_convolved)
        print()

        # OpenCV wins with a small edge but it also does more arithmetic
        # operations because of Sobel. So the comparison was not very fair.
        with sara.Timer("[SHAKTI][Halide-GPU] gradient"):
            shakti.gradient_2d_32f(video_frame_convolved, gradx, grady)
        with sara.Timer("[OPENCV] (sobel) gradient"):
            cv2.Sobel(video_frame_convolved, cv2.CV_32F, 1, 0, gradx)
            cv2.Sobel(video_frame_convolved, cv2.CV_32F, 0, 1, grady)
        print()

        # Shakti wins on CUDA-architecture by almost a factor 2.
        with sara.Timer("[SHAKTI][Halide-GPU] polar gradient"):
            shakti.polar_gradient_2d_32f(video_frame_convolved, mag, ori)
        with sara.Timer("[OPENCV] polar gradient"):
            cv2.Sobel(video_frame_convolved, cv2.CV_32F, 1, 0, gradx)
            cv2.Sobel(video_frame_convolved, cv2.CV_32F, 0, 1, grady)
            cv2.cartToPolar(gradx, grady, mag, ori)
        print()

        with sara.Timer("[SHAKTI][Halide-CPU] gray32f to rgb8"):
            mag = (mag + 1) * 0.5
            shakti.convert_gray32f_to_rgb8_cpu(mag, video_frame)

            # ori = (ori + np.pi) / (2 * np.pi)
            # shakti.convert_gray32f_to_rgb8_cpu(ori, video_frame)

        with sara.Timer("[OPENCV] gray32f to rgb8"):
            video_frame_gray8 = (video_frame_convolved * 255).astype(np.uint8)
            cv2.cvtColor(video_frame_gray8, cv2.COLOR_GRAY2RGB, video_frame)
        print()

        sara.draw_image(video_frame)

if __name__ == '__main__':
    sara.run_graphics(user_main)
