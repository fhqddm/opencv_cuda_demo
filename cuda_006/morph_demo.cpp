#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;
void morph_analysis_demo();
int main(int argc, char** argv) {
	//morph_analysis_demo();
	
	Mat gray_host = imread("../data/morph.png", 0);
	imshow("input", gray_host);

	GpuMat gray, binary, dst;
	gray.upload(gray_host);
	cuda::threshold(gray, binary, 174, 255, THRESH_BINARY_INV);
	Mat se = cv::getStructuringElement(MORPH_RECT, Size(3, 3)); 
	auto morph_filter = cuda::createMorphologyFilter(MORPH_OPEN, gray.type(), se);
	morph_filter->apply(binary, dst);
	// cuda::subtract(binary, dst, binary);

	Mat result;
	dst.download(result);
	imshow("binary", result);
	
	waitKey(0);
	return 0;
}

void morph_analysis_demo() {
	VideoCapture cap;
	cap.open("../data/vtest.avi");
	Mat frame_host, binary;
	GpuMat frame, hsv, mask;
	vector<GpuMat> mv;
	vector<GpuMat> thres(4);
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(frame_host);
		if (!ret) break;
		imshow("frame", frame_host);

		frame.upload(frame_host);
		cuda::cvtColor(frame, hsv, COLOR_BGR2HSV);
		cuda::split(hsv, mv);

		// replace inRange
		cuda::threshold(mv[0], thres[0], 35, 255, THRESH_BINARY);
		cuda::threshold(mv[0], thres[3], 77, 255, THRESH_BINARY);
		cuda::threshold(mv[1], thres[1], 43, 255, THRESH_BINARY);
		cuda::threshold(mv[2], thres[2], 46, 255, THRESH_BINARY);
		cuda::bitwise_xor(thres[0], thres[3], thres[0]);

		cuda::bitwise_and(thres[1], thres[0], mask);
		cuda::bitwise_and(mask, thres[2], mask);
		cuda::threshold(mask, mask, 66, 255, THRESH_BINARY);

		Mat se = cv::getStructuringElement(MORPH_RECT, Size(7, 7));
		auto morph_filter = cuda::createMorphologyFilter(MORPH_OPEN, mask.type(), se);
		morph_filter->apply(mask, mask);

		mask.download(binary);
		imshow("mask", binary);

		// 连通组件分析
		Mat labels = Mat::zeros(binary.size(), CV_32S);
		Mat stats, centroids;
		int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
		for (int i = 1; i < num_labels; i++) {
			int cx = centroids.at<double>(i, 0);
			int cy = centroids.at<double>(i, 1);

			int x = stats.at<int>(i, CC_STAT_LEFT);
			int y = stats.at<int>(i, CC_STAT_TOP);
			int width = stats.at<int>(i, CC_STAT_WIDTH);
			int height = stats.at<int>(i, CC_STAT_HEIGHT);
			if (width < 50 || height < 50) {
				continue;
			}
			circle(frame_host, Point(cx, cy), 2, Scalar(255, 0, 0), 2, 8, 0);
			Rect rect(x, y, width, height);
			rectangle(frame_host, rect, Scalar(0, 0, 255), 2, 8, 0);
		}

		double fps = getTickFrequency() / (getTickCount() - start);
		putText(frame_host, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("color object tracking", frame_host);
		//if (fps > 100) {
		//	break;
		//}
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

}