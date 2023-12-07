#pragma once


// HandwritingCurvature.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif


#include <windows.h>
#include <iostream>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <json/json.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

class CCurvatureOpencv
{
public:
	CCurvatureOpencv(std::string strConfigPath) {
		loadConfig(strConfigPath);
	}

	//图像增强
	void buffCuda(cv::cuda::GpuMat &gpuSrc, cv::cuda::GpuMat &gpuTar);

	//提取字符区域
	void getLetterLine(cv::cuda::GpuMat &src, cv::cuda::GpuMat &text, cv::cuda::GpuMat &tar, cv::cuda::GpuMat &tarText);

	//获得最小包围盒
	cv::RotatedRect getRotatedRect(cv::cuda::GpuMat &src);

	//转角度透视变换
	cv::cuda::GpuMat GetRotateCropImageCuda(const cv::cuda::GpuMat &srcimage, std::vector<cv::Point2f> box);

	//骨骼及较准
	void Skeleton(cv::cuda::GpuMat& gpuSrc, cv::cuda::GpuMat& gpuSrcText, cv::Mat &mtResult);

	//初步透视变化
	void remap(cv::Mat &src, cv::cuda::GpuMat &gpuTar);

private:
	void BuildPerspectiveMap(cv::Mat src);

	void loadConfig(std::string strConfigPath);

	bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

	void PerspectiveToMaps(const cv::Mat &perspective_mat, const cv::Size img_size, cv::Mat &map1, cv::Mat &map2);

	void GetRange(const cv::Mat &matSrc, int &min, int &max);

	bool createColorTable(uchar * colorTable, int Shadow, int Highlight, int OutputShadow, int OutputHighlight);

	bool createColorTables(uchar colorTables[], int min, int max);

	int adjust(cv::Mat &src, cv::Mat& dst, int min, int max);

	void buff(cv::Mat mtSrc, cv::Mat &mtTar);

	void GetRangeCuda(const cuda::GpuMat &matSrc, int &min, int &max);

private:
	cv::cuda::GpuMat gpuMatX;
	cv::cuda::GpuMat gpuMatY;

	cv::Point2f ptSrc[4];
	cv::Point2f ptTar[4];

	cv::Rect rcRoi;
};

