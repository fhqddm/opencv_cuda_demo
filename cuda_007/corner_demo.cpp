#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

RNG rng(12345);
int main(int argc, char** argv) {
	Mat image_host = imread("../data/building.jpg");
	imshow("input", image_host);

	GpuMat src, gray, corners;
	Mat dst;
	src.upload(image_host);
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	auto corner_detector = cuda::createGoodFeaturesToTrackDetector(gray.type(), 1000, 0.01, 15, 3, true);
	corner_detector->detect(gray, corners);
	corners.download(dst);
	printf("number of corners : %d \n", corners.cols);
	for (int i = 0; i < corners.cols; i++) {
		int r = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		Point2f pt = dst.at<Point2f>(0, i);
		circle(image_host, pt, 3, Scalar(b, g, r), 2, 8, 0);
	}

	imshow("corner detect result", image_host);
	waitKey(0);
	return 0;
}