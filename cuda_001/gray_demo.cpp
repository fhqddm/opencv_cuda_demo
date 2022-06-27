#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	Mat image_host = imread("../data/1.jpg");
	imshow("input", image_host);

	//Í¼Ïñ»Ò¶È»¯
	GpuMat image;
	GpuMat gray;
	image.upload(image_host);
	cuda::cvtColor(image, gray, COLOR_BGR2GRAY);

	Mat gray_host;
	gray.download(gray_host);
	imshow("gray", gray_host);
	waitKey(0);
	return 0;
}