#include "CCurvatureOpencv.h"

bool CCurvatureOpencv::polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}


void CCurvatureOpencv::PerspectiveToMaps(const cv::Mat &perspective_mat, const cv::Size img_size, cv::Mat &map1, cv::Mat &map2)
{
	cv::Mat inv_perspective(perspective_mat.inv());
	inv_perspective.convertTo(inv_perspective, CV_32FC1);

	cv::Mat xy(img_size, CV_32FC2);
	float *pxy = (float*)xy.data;
	for (int y = 0; y < img_size.height; y++)
		for (int x = 0; x < img_size.width; x++)
		{
			*pxy++ = x;
			*pxy++ = y;
		}

	cv::Mat xy_transformed;
	cv::perspectiveTransform(xy, xy_transformed, inv_perspective);

	//Prevent errors when float32 to int16
	float *pmytest = (float*)xy_transformed.data;
	for (int y = 0; y < xy_transformed.rows; y++)
		for (int x = 0; x < xy_transformed.cols; x++)
		{
			if (abs(*pmytest) > 5000) *pmytest = 5000.00;
			pmytest++;
			if (abs(*pmytest) > 5000) *pmytest = 5000.00;
			pmytest++;
		}

	// split x/y to extra maps
	assert(xy_transformed.channels() == 2);
	cv::Mat maps[2]; // map_x, map_y
	cv::split(xy_transformed, maps);

	// remap() with integer maps is faster
	cv::convertMaps(maps[0], maps[1], map1, map2, CV_32FC1);
}


void CCurvatureOpencv::GetRange(const cv::Mat &matSrc, int &min, int &max)
{
	try {
		cv::Mat matHist;
		int histSize = 256;
		float range[] = { 0, 255 };
		const float *histRanges = { range };
		cv::calcHist(&matSrc, 1, 0, cv::Mat(), matHist, 1, &histSize, &histRanges, true, false);
		int nPoints = 0;

		int nMaxLoc = 0;
		int nMax = 0;

		for (int i = 230; i > 0; i--) {
			int nValue = matHist.at<float>(i);
			if (nValue > nMax) {
				nMax = nValue;
				nMaxLoc = i;
			}
		}
		min = nMaxLoc;

		int nOffset = 0;
		int x = 0;
		int y = 0;
		int dis = 999999;
		for (int i = 255; i > nMaxLoc; i--)
		{
			int nValue = matHist.at<float>(i);
			x = i - nMaxLoc;
			y = nValue;

			int disTemp = x * x + y * y;
			if (disTemp < dis) {
				dis = disTemp;
				max = i;
			}
		}
	}
	catch (cv::Exception e) {
	}
}


bool CCurvatureOpencv::createColorTable(uchar * colorTable, int Shadow, int Highlight, int OutputShadow, int OutputHighlight)
{
	int diff = (int)(Highlight - Shadow);
	int outDiff = (int)(OutputHighlight - OutputShadow);

	if (!((Highlight <= 255 && diff <= 255 && diff >= 2) ||
		(OutputShadow <= 255 && OutputHighlight <= 255 && outDiff < 255) ||
		(!(1.0 > 9.99 && 1.0 > 0.1) && 1.0 != 1.0)))
		return false;

	double coef = 255.0 / diff;
	double outCoef = outDiff / 255.0;
	double exponent = 1.0 / 1.0;

	for (int i = 0; i < 256; i++)
	{
		int v;
		// calculate black field and white field of input level
		if (colorTable[i] <= (uchar)Shadow) {
			v = 0;
		}
		else {
			v = (int)((colorTable[i] - Shadow) * coef + 0.5);
			if (v > 255) v = 255;
		}
		// calculate midtone field of input level
		v = (int)(pow(v / 255.0, exponent) * 255.0 + 0.5);
		// calculate output level
		colorTable[i] = (uchar)(v * outCoef + OutputShadow + 0.5);
	}

	return true;
}

bool CCurvatureOpencv::createColorTables(uchar colorTables[], int min, int max)
{
	bool result = false;
	int i, j;

	for (j = 0; j < 256; j++)
		colorTables[j] = (uchar)j;

	return createColorTable(colorTables, min, max, 0, 255);
}

int CCurvatureOpencv::adjust(cv::Mat &src, cv::Mat& dst, int min, int max)
{
	cv::Mat input = src;
	if (input.empty()) {
		return -1;
	}

	dst.create(src.size(), src.type());
	cv::Mat output = dst;

	const uchar *in;
	uchar *out;
	int width = input.cols;
	int height = input.rows;

	uchar colorTables[256];
	if (!createColorTables(colorTables, min, max)) {
		return 1;
	}

	//adjust each pixel
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < height; y++) {
		in = input.ptr<uchar>(y);
		out = output.ptr<uchar>(y);
		for (int x = 0; x < width; x++) {
			*out++ = colorTables[*in++];
			//*out++ = *in++;
		}
	}
	return 0;
}

void CCurvatureOpencv::buff(cv::Mat mtSrc, cv::Mat &mtTar)
{
	cv::Mat mtSubAdjust, mtSubAdjustOut;
	int nMin, nMax;

	cv::cvtColor(mtSrc, mtSubAdjust, cv::COLOR_BGR2GRAY);
	GetRange(mtSubAdjust, nMin, nMax);//给色阶增强提供参数

	adjust(mtSubAdjust, mtSubAdjustOut, nMin, nMax);

	mtTar = mtSubAdjustOut.clone();
}

void CCurvatureOpencv::BuildPerspectiveMap(cv::Mat src)
{
	static bool bBuild = false;
	if (bBuild)
		return;

	cv::Size size(rcRoi.width, rcRoi.height);

	cv::Mat M = cv::getPerspectiveTransform(ptSrc, ptTar);
	cv::Mat x, y;
	PerspectiveToMaps(M, size, x, y);

	gpuMatX.upload(x);
	gpuMatY.upload(y);

	bBuild = true;
}

void CCurvatureOpencv::remap(cv::Mat &src, cv::cuda::GpuMat &gpuTar)
{
	BuildPerspectiveMap(src);

	cv::cuda::GpuMat gpuSrc, gpuRemap;
	gpuSrc.upload(src);
	//透视变换处理
	cv::cuda::remap(gpuSrc, gpuRemap, gpuMatX, gpuMatY, cv::INTER_LINEAR);

	//图像增强
	gpuTar = gpuRemap(rcRoi).clone();
}

void CCurvatureOpencv::Skeleton(cv::cuda::GpuMat& gpuSrc, cv::cuda::GpuMat& gpuSrcText, cv::Mat &mtResult)
{
	const int nBoundary = 300;

	cv::cuda::GpuMat gpuAll(cv::Size(gpuSrc.cols + nBoundary, gpuSrc.rows + nBoundary), CV_8UC1);
	cv::cuda::GpuMat gpuAllText(cv::Size(gpuSrc.cols + nBoundary, gpuSrc.rows + nBoundary), CV_8UC1);

	cv::cuda::GpuMat gpuThre;
	cv::cuda::threshold(gpuSrc, gpuThre, 180, 255, cv::THRESH_BINARY);

	gpuThre.copyTo(gpuAll(cv::Rect(nBoundary*0.5, nBoundary*0.5, gpuThre.cols, gpuThre.rows)));
	gpuSrcText.copyTo(gpuAllText(cv::Rect(nBoundary*0.5, nBoundary*0.5, gpuThre.cols, gpuThre.rows)));

	cv::Mat element0 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, element0);
	cv::cuda::GpuMat gpuDilate;
	dilateFilter->apply(gpuAll, gpuDilate);

	int nSize = gpuDilate.rows*gpuDilate.cols;

	cv::Mat element2 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	cv::Ptr<cv::cuda::Filter> dilateFilter2 = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, element2);
	cv::Ptr<cv::cuda::Filter> erodeFilter2 = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8U, element2);

	cv::cuda::GpuMat temp, temp2, temp3, out;
	cv::cuda::GpuMat sk(gpuDilate.size(), CV_8UC1);

	cv::Mat mtOut;
	gpuDilate.download(mtOut);

	while (true) {
		erodeFilter2->apply(gpuDilate, temp2);
		dilateFilter2->apply(temp2, temp);

		cv::cuda::subtract(gpuDilate, temp, temp3);
		cv::cuda::bitwise_or(sk, temp3, sk);

		gpuDilate = temp2.clone();

		int zero = nSize - cv::cuda::countNonZero(gpuDilate);
		if (zero == nSize)
			break;
	}
	temp.release();
	temp2.release();
	temp3.release();

	dilateFilter2->apply(sk, out);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	cv::Mat mtOut2;
	out.download(mtOut2);
	findContours(mtOut2, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(mtOut.size(), CV_8UC3);
	Mat A;
	std::vector<cv::Point> points_fitted;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours.at(i).size() > 300) {
			polynomial_curve_fit(contours.at(i), 3, A);

			int nMaxX = 0;
			int nMinX = 9999;
			for (int ii = 0; ii < contours.at(i).size(); ii++)
			{
				if (contours.at(i).at(ii).x > nMaxX) {
					nMaxX = contours.at(i).at(ii).x;
				}
				if (contours.at(i).at(ii).x < nMinX) {
					nMinX = contours.at(i).at(ii).x;
				}
			}

			for (int x = nMinX - 50; x < nMaxX + 50; x++)
			{
				double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
					A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);

				points_fitted.push_back(cv::Point(x, y));
			}
		}
		drawContours(drawing, contours, i, cv::Scalar(0, 0, 255), 1, 8, std::vector<Vec4i>(), 0, Point());
	}

	Mat mtEnd = Mat::zeros(mtOut.size(), CV_8UC1);
	cv::polylines(mtOut, points_fitted, false, cv::Scalar(255, 255, 255), 1, 8, 0);

	Mat mtTest = Mat::zeros(mtOut.size(), CV_8UC1);
	Mat mtBG = Mat::zeros(mtOut.size(), CV_8UC1);
	Mat mtSrc, mtBGSrc;
	gpuAllText.download(mtSrc);
	gpuAll.download(mtBGSrc);
	int nL, nT, nR, nB;
	nL = -1;
	nT = -1;
	nR = 999;
	nB = 999;
	for (int i = 0; i < points_fitted.size(); i++)
	{
		int x = points_fitted.at(i).x;
		int y = points_fitted.at(i).y;
		int r = 0;
		float rX = 1.1;
		for (int j = y - 100; j < y + 250; j++, r++)
		{
			mtTest.data[mtOut.cols*r + i] = mtSrc.data[mtOut.cols*j + int(x*rX)];
			mtBG.data[mtOut.cols*r + i] = mtBGSrc.data[mtOut.cols*j + int(x*rX)];
		}
	}

	cv::Mat mtThre;
	//cv::threshold(mtBG, mtThre, 1, 255, cv::THRESH_BINARY);
	findContours(mtBG, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));

	//默认获取最大的区域
	std::vector<double> areas(contours.size());
	for (size_t i = 0; i < contours.size(); ++i) {
		areas[i] = cv::contourArea(contours[i]);
	}

	// 找到面积最大的轮廓的索引
	auto max_area_index = std::distance(areas.begin(), std::max_element(areas.begin(), areas.end()));

	// 提取面积最大的轮廓
	std::vector<cv::Point> largest_contour = contours[max_area_index];

	// 提取包围盒
	cv::Rect bounding_box = cv::boundingRect(largest_contour);

	mtResult = mtTest(bounding_box).clone();
}

void CCurvatureOpencv::GetRangeCuda(const cuda::GpuMat &matSrc, int &min, int &max)
{
	try {
		cuda::GpuMat matHistGpu;
		Mat matHist;
		int histSize = 256;
		float range[] = { 0, 255 };
		const float *histRanges = { range };

		cuda::calcHist(matSrc, matHistGpu);
		matHistGpu.download(matHist);
		int nPoints = 0;

		int nMaxLoc = 0;
		int nMax = 0;

		for (int i = 255; i > 0; i--) {
			int nValue = matHist.at<int>(i);
			if (nValue > nMax) {
				nMax = nValue;
				nMaxLoc = i;
			}
		}
		min = nMaxLoc;

		int nOffset = 0;
		int x = 0;
		int y = 0;
		int dis = 999999;
		for (int i = 255; i > nMaxLoc; i--)
		{
			int nValue = matHist.at<int>(i);
			x = i - nMaxLoc;
			y = nValue;

			int disTemp = x * x + y * y;
			if (disTemp < dis) {
				dis = disTemp;
				max = i;
			}
		}
	}
	catch (cv::Exception e) {
	}
}

void CCurvatureOpencv::buffCuda(cv::cuda::GpuMat &gpuSrc, cv::cuda::GpuMat &gpuTar)
{
	int nMin, nMax;

	GetRangeCuda(gpuSrc, nMin, nMax);

	cv::Mat mtSubAdjust, mtSubAdjustOut;
	gpuSrc.download(mtSubAdjust);
	adjust(mtSubAdjust, mtSubAdjustOut, nMin, nMax);

	gpuTar.upload(mtSubAdjustOut);
}

void CCurvatureOpencv::getLetterLine(cv::cuda::GpuMat &src, cv::cuda::GpuMat &text, cv::cuda::GpuMat &tar, cv::cuda::GpuMat &tarText)
{
	//提取字符区域
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	cv::cuda::GpuMat gpuThre;
	cv::cuda::threshold(src, gpuThre, 125, 255, cv::THRESH_BINARY);

	cv::Mat elem = getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31)); //具体数值取决于字符大小
	cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, elem);
	cv::cuda::GpuMat gpuDilate;
	dilateFilter->apply(gpuThre, gpuDilate);

	cv::Mat mtSrc;
	gpuDilate.download(mtSrc);

	findContours(mtSrc, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	std::vector<Rect> boundRect(contours.size());

	//默认获取最大的区域
	std::vector<double> areas(contours.size());
	for (size_t i = 0; i < contours.size(); ++i) {
		areas[i] = cv::contourArea(contours[i]);
	}

	// 找到面积最大的轮廓的索引
	auto max_area_index = std::distance(areas.begin(), std::max_element(areas.begin(), areas.end()));

	// 提取面积最大的轮廓
	std::vector<cv::Point> largest_contour = contours[max_area_index];

	// 提取包围盒
	cv::Rect bounding_box = cv::boundingRect(largest_contour);

	tar = gpuDilate(bounding_box).clone();
	tarText = text(bounding_box).clone();
}

cv::RotatedRect CCurvatureOpencv::getRotatedRect(cv::cuda::GpuMat &src)
{
	//提取字符区域
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	cv::Mat mtSrc;
	src.download(mtSrc);
	findContours(mtSrc, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	std::vector<Rect> boundRect(contours.size());

	//默认获取最大的区域
	std::vector<double> areas(contours.size());
	for (size_t i = 0; i < contours.size(); ++i) {
		areas[i] = cv::contourArea(contours[i]);
	}

	// 找到面积最大的轮廓的索引
	auto max_area_index = std::distance(areas.begin(), std::max_element(areas.begin(), areas.end()));

	// 提取面积最大的轮廓
	std::vector<cv::Point> largest_contour = contours[max_area_index];

	// 提取包围盒
	cv::RotatedRect boundingBox = cv::minAreaRect(largest_contour);

	return boundingBox;
}

cv::cuda::GpuMat CCurvatureOpencv::GetRotateCropImageCuda(const cv::cuda::GpuMat &srcimage, std::vector<cv::Point2f> box)
{
	std::vector<cv::Point2f> points = box;

	int x_collect[4] = { box[0].x, box[1].x, box[2].x, box[3].x };
	int y_collect[4] = { box[0].y, box[1].y, box[2].y, box[3].y };
	int left = int(*std::min_element(x_collect, x_collect + 4));
	int right = int(*std::max_element(x_collect, x_collect + 4));
	int top = int(*std::min_element(y_collect, y_collect + 4));
	int bottom = int(*std::max_element(y_collect, y_collect + 4));

	if (top < 0)
		top = 0;
	if (left < 0)
		left = 0;

	int width = right - left;
	int height = bottom - top;
	if (left + width >= srcimage.cols)
	{
		width = srcimage.cols - left - 1;
	}
	if (height + top >= srcimage.rows)
	{
		height = srcimage.rows - top - 1;
	}

	if (width <= 0 || height <= 0) {
		Mat mat = Mat::zeros(0, 0, srcimage.type());
		return cv::cuda::GpuMat();
	}

	int centL = left;
	int centT = top;
	int centB = bottom;
	int centR = right;
	int centW = width;
	int centH = height;

	cv::cuda::GpuMat img_crop;
	srcimage(cv::Rect(left, top, width, height)).copyTo(img_crop);

	for (int i = 0; i < points.size(); i++) {
		points[i].x -= left;
		points[i].y -= top;
	}

	int img_crop_width = int(sqrt(pow(points[0].x - points[1].x, 2) +
		pow(points[0].y - points[1].y, 2)));
	int img_crop_height = int(sqrt(pow(points[0].x - points[3].x, 2) +
		pow(points[0].y - points[3].y, 2)));

	cv::Point2f pts_std[4];
	pts_std[0] = cv::Point2f(0., 0.);
	pts_std[1] = cv::Point2f(img_crop_width, 0.);
	pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
	pts_std[3] = cv::Point2f(0.f, img_crop_height);

	cv::Point2f pointsf[4];
	pointsf[0] = points[0];
	pointsf[1] = points[1];
	pointsf[2] = points[2];
	pointsf[3] = points[3];

	cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

	cv::cuda::GpuMat dst_img;

	cv::cuda::warpPerspective(img_crop, dst_img, M,
		cv::Size(img_crop_width, img_crop_height),
		cv::BORDER_REPLICATE);

	return dst_img;
}


void CCurvatureOpencv::loadConfig(std::string strConfigPath)
{
	Json::Reader reader;
	Json::Value root;

	ifstream in(strConfigPath.c_str(), ios::binary);

	if (reader.parse(in, root))
	{
		ptSrc[0].x = root["p1-1"][0].asInt();
		ptSrc[0].y = root["p1-1"][1].asInt();

		ptSrc[1].x = root["p1-2"][0].asInt();
		ptSrc[1].y = root["p1-2"][1].asInt();

		ptSrc[2].x = root["p1-3"][0].asInt();
		ptSrc[2].y = root["p1-3"][1].asInt();

		ptSrc[3].x = root["p1-4"][0].asInt();
		ptSrc[3].y = root["p1-4"][1].asInt();

		int ax = ptSrc[0].x;
		int ay = ptSrc[0].y;
		int cx = ptSrc[2].x;
		int cy = ptSrc[2].y;
		int dx = ptSrc[3].x;
		int dy = ptSrc[3].y;
		int ptTarw = sqrt(pow((dx - cx), 2) + pow((dy - cy), 2));
		int ptTarh = sqrt(pow((ax - cx), 2) + pow((ay - cy), 2));

		ptTar[0].x = 0;
		ptTar[0].y = 0;
		ptTar[1] = Point(ptTarw, 0);
		ptTar[2] = Point(0, ptTarh);
		ptTar[3] = Point(ptTarw, ptTarh);

		rcRoi.x = 0;
		rcRoi.y = 0;
		rcRoi.width = ptTarw;
		rcRoi.height = ptTarh;
	}
}
