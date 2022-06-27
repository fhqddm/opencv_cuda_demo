#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

void optical_flow_demo();
void background_demo();
int main(int argc, char** argv) {
	optical_flow_demo();
	waitKey(0);
	return 0;
}

void background_demo() {
	VideoCapture cap;
	cap.open("../data/vtest.avi");
	auto mog = cuda::createBackgroundSubtractorMOG2();
	Mat frame;
	GpuMat d_frame, d_fgmask, d_bgimg;
	Mat fg_mask, bgimg, fgimg;
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("background", WINDOW_AUTOSIZE);
	namedWindow("mask", WINDOW_AUTOSIZE);
	Mat se = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(frame);
		if (!ret) break;

		// 背景分析
		d_frame.upload(frame);
		mog->apply(d_frame, d_fgmask);
		mog->getBackgroundImage(d_bgimg);

		// 形态学操作
		auto morph_filter = cuda::createMorphologyFilter(MORPH_OPEN, d_fgmask.type(), se);
		morph_filter->apply(d_fgmask, d_fgmask);

		// download from GPU Mat
		d_bgimg.download(bgimg);
		d_fgmask.download(fg_mask);

		// 计算FPS
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(frame, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);

		imshow("input", frame);
		imshow("background", bgimg);
		imshow("mask", fg_mask);
		
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return;
}

void optical_flow_demo() {
	VideoCapture cap;
	cap.open("../data/vtest.avi");
	auto farn = cuda::FarnebackOpticalFlow::create();
	Mat f, pf;
	cap.read(pf);

	GpuMat frame, gray, preFrame, preGray;
	preFrame.upload(pf);
	cuda::cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	Mat hsv = Mat::zeros(preFrame.size(), preFrame.type());

	GpuMat flow;
	vector<Mat> mv;
	split(hsv, mv);
	GpuMat gMat, gAng;
	Mat mag = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ang = Mat::zeros(hsv.size(), CV_32FC1);

	gMat.upload(mag);
	gAng.upload(ang);

	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("optical flow demo", WINDOW_AUTOSIZE);

	Mat se = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(f);
		if (!ret) break;

		// 光流分析
		frame.upload(f);
		cuda::cvtColor(frame, gray, COLOR_BGR2GRAY);
		farn->calc(preGray, gray, flow);

		// 坐标转换
		vector<GpuMat> mm;
		cuda::split(flow, mm);
		cuda::cartToPolar(mm[0], mm[1], gMat, gAng);
		cuda::normalize(gMat, gMat, 0, 255, NORM_MINMAX, CV_32FC1);
		gMat.download(mag);
		gAng.download(ang);

		// 显示
		ang = ang * 180 / CV_PI / 2.0;
		convertScaleAbs(mag, mag);
		convertScaleAbs(ang, ang);
		mv[0] = ang;
		mv[1] = Scalar(255);
		mv[2] = mag;
		merge(mv, hsv);
		Mat bgr;
		cv::cvtColor(hsv, bgr, COLOR_HSV2BGR);


		// 计算FPS
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(f, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		gray.copyTo(preGray);
		imshow("input", f);
		imshow("optical flow demo", bgr);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return;
}