# HandwritingCurvature

本代码实现对手写板号，或其他带小幅弧度的字体进行矫正的功能。

开发环境如下：
语言：C++
编译环境：Visual Studio 2017 X64
依赖库：
Opencv 4.0+（with CUDA)
Jsoncpp 1.9.2
cuda 10以上，需与Opencv编译所依赖的CUDA版本一致

此外，代码中涉及字符检测模型的部分，采用的是PaddleOCR 2.X，因文件较大，编译依赖环境较多，请自行编译准备，链接如下：
https://github.com/PaddlePaddle/PaddleOCR

