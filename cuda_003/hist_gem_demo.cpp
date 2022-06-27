#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	Mat image_host = imread("../data/chicky_512.png");
	imshow("input", image_host);

	GpuMat image(image_host);
	vector<GpuMat> mv;
	GpuMat hist, hsv;
	cuda::split(image, mv);
	cuda::calcHist(mv[2], hist);

	Mat hist_host;
	hist.download(hist_host);
	for (int i = 0; i < hist_host.cols; i++) {
		int pv = hist_host.at<int>(0, i);
		printf("total number : %d, of the pixel value : %d \n", pv, i);
	}

	cuda::cvtColor(image, hsv, COLOR_BGR2HSV);
	cuda::split(hsv, mv);
	cuda::equalizeHist(mv[2], mv[2]);
	cuda::merge(mv, hsv);
	cuda::cvtColor(hsv, image, COLOR_HSV2BGR);

	Mat result;
	image.download(result);
	imshow("eq-demo result", result);

	// resize and rotate
	GpuMat dst;
	cuda::resize(image, dst, Size(0, 0), 2, 2, INTER_CUBIC);
	dst.download(result);
	imshow("resize result", result);

	int cx = image.cols / 2;
	int cy = image.rows / 2;
	Mat M = getRotationMatrix2D(Point(cx, cy), 45, 1.0);
	cuda::warpAffine(image, dst, M, image.size());
	dst.download(result);
	imshow("rotate result", result);

	waitKey(0);
	return 0;
}