#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;
void cpu_demo();
int main(int argc, char** argv) {
	//Mat image_host = imread("../data/lena.jpg");
	//imshow("input", image_host);
	// cpu_demo();
	VideoCapture cap;
	cap.open("../data/Megamind.avi");
	Mat frame,  result;
	GpuMat image;
	GpuMat dst;
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(frame);
		if (!ret) break;
		image.upload(frame);
		cuda::cvtColor(image, image, COLOR_BGR2BGRA);
		cuda::bilateralFilter(image, dst, 0, 100, 10);
		// cuda::meanShiftFiltering(image, dst, 7, 50);
		dst.download(result);
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(result, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("GPU-demo", result);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}

void cpu_demo() {
	VideoCapture cap;
	cap.open("../data/Megamind.avi");
	Mat frame, result;
	while (true) {
		int64 start = getTickCount();
		cap.read(frame);
		cv::bilateralFilter(frame, result, 0, 100, 14, 4);
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(result, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("CPU-demo", result);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return;
}