// HandwritingCurvature.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif

#include "CCurvatureOpencv.h"

#include <windows.h>
#include <iostream>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <libloaderapi.h>

using namespace cv;
using namespace std;

bool detArea(cv::cuda::GpuMat *src, cv::cuda::GpuMat *tar)
{
	HMODULE hWnd = LoadLibraryA("ppocr.dll");

	typedef bool(_cdecl *PFUNC)(cv::cuda::GpuMat * src, cv::cuda::GpuMat *tar);
	PFUNC pFunc = (PFUNC)GetProcAddress(hWnd, "detCuda");

	if (pFunc != NULL) {
		return pFunc(src, tar);
	}
	return false;
}

int main()
{
	CCurvatureOpencv curvatureOpencv("config.txt");

	//读取处理图像
	cv::Mat src = cv::imread("D:\\test.jpg",0);

	//透视变换矫正
	cv::cuda::GpuMat gpuTar;
	curvatureOpencv.remap(src, gpuTar);

	//图像增强
	cv::cuda::GpuMat gpuBuff;
	curvatureOpencv.buffCuda(gpuTar, gpuBuff);

	//获取检测区域 调用PaddleOCR的DET检测模型，返回的格式为OCR_DET.cpp 中 DBDetector::Run 里 bit_map
	//由于文件较大，对依赖库版本要求较高，这里不给出源代码，如有需要请自行编译
	cv::cuda::GpuMat gpuDet(cv::Size(gpuBuff.cols, gpuBuff.rows), CV_8UC1);
	detArea(&gpuBuff, &gpuDet);

	//提取字符区域
	cv::cuda::GpuMat gpuLetterArea, gpuText;
	curvatureOpencv.getLetterLine(gpuDet, gpuBuff, gpuLetterArea, gpuText);

	//获取最小外接矩形
	cv::RotatedRect rRect = curvatureOpencv.getRotatedRect(gpuLetterArea);
	cv::Point2f vertex[4];
	rRect.points(vertex);
	std::vector<cv::Point2f> vec;
	vec.push_back(vertex[2]);
	vec.push_back(vertex[3]);
	vec.push_back(vertex[0]);
	vec.push_back(vertex[1]);

	//微调方向
	cv::cuda::GpuMat gpuLetterLine = curvatureOpencv.GetRotateCropImageCuda(gpuLetterArea, vec);
	cv::cuda::GpuMat gpuLetter = curvatureOpencv.GetRotateCropImageCuda(gpuText, vec);

	//提取骨骼及转换
	cv::Mat mtOut;
	curvatureOpencv.Skeleton(gpuLetterLine, gpuLetter, mtOut);

	cv::imwrite("out.jpg", mtOut);

	return 1;
}
