# Yolo Windows and Linux version

## About this fork:
This is forked from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) for Windows(mainly) and Linux develop environment, and set upstream from [pjreddie/darknet](https://github.com/pjreddie/darknet) for weights data structor which has been changed.

It also contains python wrapper and simple usage in [python directory](https://github.com/sharowyeh/darknet/tree/master-pjreddie/python) for libdarknet.so or libdarknet.dll (which is compiled from gcc or visual studio)

## How to compile on Windows:

1. It required **Visual Studio 2015** or later version with Platform Toolset v140 building source code, or visit [Visual Studio Downloads](https://www.visualstudio.com/downloads/) for more IDE information. The solution files of source code located at **build\darknet\**, use **Release** and **x64** building configuration

2. Download **OpenCV 3.0** and follow [How to set building environment variables with Visual Studio](https://docs.opencv.org/3.3.0/d6/d8a/tutorial_windows_visual_studio_Opencv.html), and variable path should be target to **{_OpenCV Install Path_}\build\x64\vc14**

    - Copy files **opencv_world{_ver_}.dll** and **opencv_ffmpeg{*ver*}\_64.dll** from **{_OpenCV Install Path_}\build\x64\vc14\bin** to compile output directory which **darknet.exe** located.

    - If you have **OpenCV 2.4.13** instead of 3.0, check **_OPENCV_DIR_** system environment variable is correct

3. (Optional but recommanded) Download **CUDA 9.1** and follow [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), installer will automatically add **_CUDA_PATH_** to system environment variables, and create CUDA related macros in Visual Studio IDE for project properties

   - If you have other version of **CUDA (not 9.1)**, check **_CUDA_PATH_** system environment variable is targeting to CUDA installation root directory, **{CudaToolkitIncludeDir}** and **{CudaToolkitLibDir}** macro in Visual Studio IDE have valid path for inblude and lib paths

   - If you want to build with CUDNN to speed up compiler, download **cuDNN 7.0 for CUDA 9.1** from [here](https://developer.nvidia.com/cudnn) and follow [installation guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). In Visual Studio Project Properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, add or remove **CUDNN** predefined keyword

4. If you **don't have GPU**, use **darknet_no_gpu.sln** instead

5. If you have different installation location of OpenCV or CUDA for Visual Studio:

    - add include directory to Project Properties -> C/C++ -> General -> Additional Include Directories

    - add lib directory to Project Properties -> Linker -> General -> Additional Library Directories
