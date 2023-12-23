N.B.: the notes may not be entirely accurate, so please refer to the Dockerfile:
`sara/docker/Dockerfile.pytorch-cuda` as well.

- Clean up all CUDA packages
  ```
  sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
  ```
  Ref: [https://stackoverflow.com/questions/56431461/how-to-remove-cuda-completely-from-ubuntu]

- Unhold all packages:
  ```
  sudo apt-mark unhold libcudnn8 libcudnn8-dev tensorrt-dev
  ```

- Install the CUDA deb (network) package as detailed in the website
  ```
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda
  ```
- Update the cuda-keyring package
  ```
  sudo apt update -y
  sudo apt upgrade
  ```

- Install the specific CUDA version:
  ```
  sudo apt install cuda-12-0
  ```

- Install the specific cuDNN version:
  ```
  apt list -a libcudnn8
  sudo apt install libcudnn8=8.8.0.121-1+cuda12.0
  sudo apt install libcudnn8-dev=8.8.0.121-1+cuda12.0

  # Lock these packages to these versions to prevent APT from updating them.
  sudo apt-mark hold libcudnn8 libcudnn8-dev tensorrt-dev
  ```

  1. First check on the TensorRT which cuDNN version we can install:
  2. Then follow the instructions on this page:
     [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install]

- Install the specific TensorRT version:
  ```
  apt list -a tensorrt-dev
  version="8.6.1.6-1+cuda12.0"
  sudo apt install tensorrt-dev=${version}

  # Lock these packages to these versions to prevent APT from updating them.
  sudo apt-mark hold tensorrt-dev
  ```

  More info on this page:
  [https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing]
