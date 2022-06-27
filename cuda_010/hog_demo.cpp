#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	VideoCapture cap;
	cap.open("../data/vtest.avi");

	Mat f;
	GpuMat frame, gray;
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("People Detector Demo", WINDOW_AUTOSIZE);

	// ¥¥Ω®ºÏ≤‚∆˜
	auto hog = cuda::HOG::create();
	hog->setSVMDetector(hog->getDefaultPeopleDetector());
	vector<Rect> objects;
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(f);
		if (!ret) break;
		imshow("input", f);

		// HOG detector
		frame.upload(f);
		cuda::cvtColor(frame, gray, COLOR_BGR2GRAY);
		hog->detectMultiScale(gray, objects);

		// ªÊ÷∆ºÏ≤‚
		for (int i = 0; i < objects.size(); i++) {
			rectangle(f, objects[i], Scalar(0, 0, 255), 2, 8, 0);
		}

		// º∆À„FPS
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(f, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("People Detector Demo", f);

		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}

	cv::waitKey(0);
	return 0;
}