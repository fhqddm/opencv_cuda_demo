#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	Mat src1_host = imread("../data/WindowsLogo.jpg");
	Mat src2_host = imread("../data/LinuxLogo.jpg");
	imshow("input1", src1_host);
	imshow("input2", src2_host);

	// GPU 对象
	GpuMat src1, src2, dst;
	src1.upload(src1_host);
	src2.upload(src2_host);
	cuda::add(src1, src2, dst);
	// cuda::subtract(src1, src2, dst);
	// cuda::multiply(src1, src2, dst);

	Mat result;
	dst.download(result);
	imshow("result", result);

	// 权重加减
	Mat src_host = imread("../data/building.jpg");
	imshow("input", src_host);
	GpuMat src;
	src.upload(src_host);
	GpuMat blank = GpuMat(src.size(), src.type());
	cuda::addWeighted(src1, 0.5, src2, 0.5, 0, dst);
	cuda::bitwise_not(dst, dst);
	dst.download(result);
	imshow("weigthed add", result);

	GpuMat hsv, rgb, gray, YCrCb;
	cuda::cvtColor(src, hsv, COLOR_BGR2HSV);
	cuda::cvtColor(src, rgb, COLOR_BGR2RGB);
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	cuda::cvtColor(src, YCrCb, COLOR_BGR2YCrCb);

	Mat hsv_host, rgb_host, gray_host, YCrCb_host;
	hsv.download(hsv_host);
	rgb.download(rgb_host);
	gray.download(gray_host);
	YCrCb.download(YCrCb_host);

	imshow("hsv", hsv_host);
	imshow("rgb", rgb_host);
	imshow("gray", gray_host);
	imshow("YCrCb", YCrCb_host);

	waitKey(0);
	return 0;
}