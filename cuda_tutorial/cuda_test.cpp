#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
	cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = getCudaEnabledDeviceCount();
	printf("GPU Device count %d \n", count);
	return 0;
}