#include <iostream>	
#include <opencv2\opencv.hpp>

#define _CRT_SECURE_NO_WARNINGS

#ifdef _DEBUG
#pragma comment (lib, "opencv_world320d.lib")
#else
#pragma comment (lib, "opencv_world320.lib")
#endif // _DEBUG

#define PI 3.1415926

using namespace cv;

class LineFinder {
private:

	// original image
	cv::Mat img;

	// vector containing the end points 
	// of the detected lines
	std::vector<cv::Vec4i> lines;

	// accumulator resolution parameters
	double deltaRho;
	double deltaTheta;

	// minimum number of votes that a line 
	// must receive before being considered
	int minVote;

	// min length for a line
	double minLength;

	// max allowed gap along the line
	double maxGap;

	// distance to shift the drawn lines down when using a ROI
	int shift;

public:

	// Default accumulator resolution is 1 pixel by 1 degree
	// no gap, no mimimum length
	LineFinder() : deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.) {}

	// Set the resolution of the accumulator
	void setAccResolution(double dRho, double dTheta) {

		deltaRho = dRho;
		deltaTheta = dTheta;
	}

	// Set the minimum number of votes
	void setMinVote(int minv) {

		minVote = minv;
	}

	// Set line length and gap
	void setLineLengthAndGap(double length, double gap) {

		minLength = length;
		maxGap = gap;
	}

	// set image shift
	void setShift(int imgShift) {

		shift = imgShift;
	}

	// Apply probabilistic Hough Transform
	std::vector<cv::Vec4i> findLines(cv::Mat& binary) {

		lines.clear();
		cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);

		return lines;
	}

	// Draw the detected lines on an image
	void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(255)) {

		// Draw the lines
		std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();

		while (it2 != lines.end()) {

			cv::Point pt1((*it2)[0], (*it2)[1] + shift);
			cv::Point pt2((*it2)[2], (*it2)[3] + shift);

			cv::line(image, pt1, pt2, color, 6);
			std::cout << " HoughP line: (" << pt1 << "," << pt2 << ")\n";
			++it2;
		}
	}

	// Eliminates lines that do not have an orientation equals to
	// the ones specified in the input matrix of orientations
	// At least the given percentage of pixels on the line must 
	// be within plus or minus delta of the corresponding orientation
	std::vector<cv::Vec4i> removeLinesOfInconsistentOrientations(
		const cv::Mat &orientations, double percentage, double delta) {

		std::vector<cv::Vec4i>::iterator it = lines.begin();

		// check all lines
		while (it != lines.end()) {

			// end points
			int x1 = (*it)[0];
			int y1 = (*it)[1];
			int x2 = (*it)[2];
			int y2 = (*it)[3];

			// line orientation + 90o to get the parallel line
			double ori1 = atan2(static_cast<double>(y1 - y2), static_cast<double>(x1 - x2)) + PI / 2;
			if (ori1>PI) ori1 = ori1 - 2 * PI;

			double ori2 = atan2(static_cast<double>(y2 - y1), static_cast<double>(x2 - x1)) + PI / 2;
			if (ori2>PI) ori2 = ori2 - 2 * PI;

			// for all points on the line
			cv::LineIterator lit(orientations, cv::Point(x1, y1), cv::Point(x2, y2));
			int i, count = 0;
			for (i = 0, count = 0; i < lit.count; i++, ++lit) {

				float ori = *(reinterpret_cast<float *>(*lit));

				// is line orientation similar to gradient orientation ?
				if (std::min(fabs(ori - ori1), fabs(ori - ori2))<delta)
					count++;

			}

			double consistency = count / static_cast<double>(i);

			// set to zero lines of inconsistent orientation
			if (consistency < percentage) {

				(*it)[0] = (*it)[1] = (*it)[2] = (*it)[3] = 0;

			}

			++it;
		}

		return lines;
	}
};

int main()
{
	int scale = 1.5;
	int delta = 0;
	int ddepth = CV_16S;
	int key_pressed = 0;
	
	VideoCapture capture("tracking1.mp4");

	namedWindow("Processed Video", CV_WINDOW_KEEPRATIO);
	//namedWindow("Video", CV_WINDOW_KEEPRATIO);

	// Show video information
	int width = 0, height = 0, fps = 0, fourcc = 0;
	width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
	fourcc = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));

	if (!capture.isOpened())
		return -1;
	
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	std::cout << "Frame Size = " << dWidth << "x" << dHeight << std::endl;

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	Mat lane_tracking;
	Mat gray;
	Mat eq;
	Mat Gaussian;
	Mat medianFilter;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat grad;
	Mat contours;

	LineFinder ld; // 인스턴스 생성

	while (key_pressed != 27) {		
		capture >> lane_tracking;		//원본 추출
		cvtColor(lane_tracking, gray, CV_BGR2GRAY, BORDER_DEFAULT);	//그레이 스케일링
		//추후에 iteration으로 한 작업과 시간비교해서 큰 차이가 없으면 iteration으로 교체(노란색 강조)

		equalizeHist(gray, eq);			// 영상 평활화		

		GaussianBlur(gray, Gaussian, Size(3, 3), 1.5);	//가우시안 필터(잡음 제거)
		//medianBlur(eq, medianFilter, 1);				//미디안 필터
		
		//샤흐 X 성분
		//Sobel(Gaussian, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		Scharr(Gaussian, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		//샤흐 Y 성분
		//Sobel(Gaussian, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		Scharr(Gaussian, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
				
		//전체 합성 성분(추정치)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		Rect roi(0, grad.cols / 3, grad.cols - 1, grad.rows - grad.cols / 3);// 관심영역 설정
		Mat imgROI = grad(roi);

		Canny(imgROI, contours, 350, 500, 3);

		imshow("Processed Video", contours);
		//imshow("Video", eq);

		key_pressed = waitKey(5);
	}


}