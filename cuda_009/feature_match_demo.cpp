#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	// cpu data
	Mat h_object_image = imread("../data/box.png", 0); // with a leather target image
	Mat h_scene_image = imread("../data/box_in_scene.png", 0); // scene image,

	// gpu data
	cuda::GpuMat d_object_image;
	cuda::GpuMat d_scene_image;

	vector< KeyPoint > h_keypoints_scene, h_keypoints_object; // CPU key points
	cuda::GpuMat d_descriptors_scene, d_descriptors_object;   // GPU descriptor

	// Image CPU uploaded to GPU
	d_object_image.upload(h_object_image);
	d_scene_image.upload(h_scene_image);

	// 对象检测
	auto orb = cuda::ORB::create();
	// Detect feature points and extract corresponding descriptors
	orb->detectAndCompute(d_object_image, cuda::GpuMat(), h_keypoints_object, d_descriptors_object);
	orb->detectAndCompute(d_scene_image, cuda::GpuMat(), h_keypoints_scene, d_descriptors_scene);

	// Brute Force Violence Matcher
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	vector< vector< DMatch> > d_matches;
	matcher->knnMatch(d_descriptors_object, d_descriptors_scene, d_matches, 2);

	std::cout << "match size:" << d_matches.size() << endl;
	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(h_keypoints_object.size() - 1, d_matches.size()); k++)
	{
		if ((d_matches[k][0].distance < 0.9*(d_matches[k][1].distance)) &&
			((int)d_matches[k].size() <= 2 && (int)d_matches[k].size()>0))
		{
			good_matches.push_back(d_matches[k][0]);
		}
	}
	std::cout << "size:" << good_matches.size() << endl;

	// 绘制匹配点对
	Mat h_image_result;
	drawMatches(h_object_image, h_keypoints_object, h_scene_image, h_keypoints_scene,
		good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::DEFAULT);

	// Find the image pixel 2d coordinates corresponding to the matching point pair
	std::vector<Point2f> object;
	std::vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++)
	{
		object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
	}

	// 计算单应性矩阵
	Mat Homo = findHomography(object, scene, RANSAC);
	std::vector<Point2f> corners(4); // four corners of the image
	std::vector<Point2f> scene_corners(4);

	// 透视变换
	corners[0] = Point(0, 0);
	corners[1] = Point(h_object_image.cols, 0);
	corners[2] = Point(h_object_image.cols, h_object_image.rows);
	corners[3] = Point(0, h_object_image.rows);
	perspectiveTransform(corners, scene_corners, Homo);

	// 绘制对象
	line(h_image_result, scene_corners[0] + Point2f(h_object_image.cols, 0), scene_corners[1] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[1] + Point2f(h_object_image.cols, 0), scene_corners[2] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[2] + Point2f(h_object_image.cols, 0), scene_corners[3] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[3] + Point2f(h_object_image.cols, 0), scene_corners[0] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);

	imshow("Good Matches & Object detection", h_image_result);

	waitKey(0);
	return 0;
}