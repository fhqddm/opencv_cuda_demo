#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main(int argc, char** argv) {
    Mat image_host = imread("../data/lena.jpg");
    imshow("input", image_host);

    GpuMat image, d_result3x3, d_result5x5, d_result9x9;
    image.upload(image_host);
    cuda::cvtColor(image, image, COLOR_BGR2BGRA);

    // create box filter
    // auto filter_3x3 = cuda::createBoxFilter(image.type(), image.type(), Size(3, 3), Point(-1, -1));
    // auto filter_5x5 = cuda::createBoxFilter(image.type(), image.type(), Size(5, 5), Point(-1, -1));
    // auto filter_9x9 = cuda::createBoxFilter(image.type(), image.type(), Size(9, 9), Point(-1, -1));

    // create gaussian filter
    auto filter_3x3 = cuda::createGaussianFilter(image.type(), image.type(), Size(5, 5), 5);
    auto filter_5x5 = cuda::createGaussianFilter(image.type(), image.type(), Size(15, 15), 15);
    auto filter_9x9 = cuda::createGaussianFilter(image.type(), image.type(), Size(25, 25), 25);

    // apply them
    filter_3x3->apply(image, d_result3x3);
    filter_5x5->apply(image, d_result5x5);
    filter_9x9->apply(image, d_result9x9);

    // image gradient
    auto sobel_dx = cuda::createSobelFilter(image.type(), image.type(), 1, 0, 3);
    auto sobel_dy = cuda::createSobelFilter(image.type(), image.type(), 0, 1, 3);

    GpuMat grad_x, grad_y, gradxy;
    sobel_dx->apply(image, grad_x);
    sobel_dy->apply(image, grad_y);
    cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradxy);

    Mat grad_host;
    gradxy.download(grad_host);
    imshow("gradient demo", grad_host);

    // 边缘提取
    GpuMat gray, edges;
    cuda::cvtColor(image, gray, COLOR_BGRA2GRAY);
    // auto edge_detector = cuda::createCannyEdgeDetector(50, 150, 3, true);
    // edge_detector->detect(gray, edges);
    auto laplacian_filter = cuda::createLaplacianFilter(gray.type(), gray.type(), 3, 1.0);
    laplacian_filter->apply(gray, edges);
    Mat edges_host;
    edges.download(edges_host);
    imshow("Canny Edge Demo", edges_host);

    // 下载数据
    Mat result3, result5, result9;
    d_result3x3.download(result3);
    d_result5x5.download(result5);
    d_result9x9.download(result9);

    // 显示数据
    imshow("filter3x3 result", result3);
    imshow("filter5x5 result", result5);
    imshow("filter9x9 result", result9);

    waitKey(0);
    return 0;
}
