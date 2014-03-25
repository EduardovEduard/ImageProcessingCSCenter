#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace cv;

const std::string LENA_ORIGINAL_GRAY = "./images/lena_gray_512.tif";
const std::string LENA_NOISED_GRAY = "./images/lena_diagonal.jpg";

const std::string LENA_ORIGINAL = "./images/lena_color_512.tif";
const std::string LENA_NOISED = "./images/lena_color_512-noise.tif";

double mse(const Mat& im1, const Mat& im2)
{
  MatConstIterator_<uchar> it1 = im1.begin<uchar>();
  MatConstIterator_<uchar> it2 = im2.begin<uchar>();

  int n = 0;
  double result = 0;
  for (; it1 != im1.end<uchar>(); it1++, it2++)
  {
    result += std::pow((*it1 - *it2), 2);
    ++n;
  }

  return result / n;
}
Mat original, noised, filtered;

Mat prepareForDFT(const Mat& image)
{
  Mat result;
  int n = getOptimalDFTSize(image.cols);
  int m = getOptimalDFTSize(image.rows);
  copyMakeBorder(image, result, 0, m - image.rows, 0,n - image.cols,
                 BORDER_CONSTANT, Scalar::all(0));
  return result;
}

void flipQuadrants(Mat& im)
{
  int cx = im.cols/2;
  int cy = im.rows/2;

  Mat q0(im, Rect(0, 0, cx, cy));
  Mat q1(im, Rect(cx, 0, cx, cy));
  Mat q2(im, Rect(0, cy, cx, cy));
  Mat q3(im, Rect(cx, cy, cx, cy));

  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

std::vector<Point2d> locatePeaks(const Mat& peaks)
{
  assert(peaks.type() == CV_32F);

  std::vector<Point2d> result;
  for (int i = 0; i < peaks.rows; ++i)
  {
    const float* ptr = peaks.ptr<float>(i);
    for (int j = 0; j < peaks.cols; ++j)
    {
      if (ptr[j] > 0 && j != peaks.cols / 2)
      {
        result.push_back(Point2d(j, i));
      }
    }
  }

  return result;
}

void task3_5()
{
  original = imread(LENA_ORIGINAL_GRAY, CV_LOAD_IMAGE_GRAYSCALE);
  noised = imread(LENA_NOISED_GRAY, CV_LOAD_IMAGE_GRAYSCALE);

  Mat padded = prepareForDFT(noised);

  Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };

  Mat complex;
  merge(planes, 2, complex);

  dft(complex, complex);

  split(complex, planes);

  Mat mag;
  magnitude(planes[0], planes[1], mag);

  mag += Scalar::all(1);
  log(mag, mag);

  mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
  flipQuadrants(mag);

  normalize(mag, mag, 0, 1, CV_MINMAX);

  Mat peaks(mag.size(), CV_32F);
  threshold(mag, peaks, 0.89, 1, THRESH_BINARY);

  std::vector<Point2d> points = locatePeaks(peaks);

  Mat filter = Mat::ones(mag.size(), CV_32F);
  for (std::vector<Point2d>::iterator it = points.begin(); it != points.end(); ++it)
  {
    const Point2d& point = *it;
    line(filter, Point(point.x, 0), Point(point.x, mag.rows), Scalar::all(0), 30);
  }

  flipQuadrants(filter);

  Mat filterPlanes[] = { filter, Mat::zeros(filter.size(), CV_32F) };
  merge(filterPlanes, 2, filter);

  Mat result;
  mulSpectrums(complex, filter, result, DFT_REAL_OUTPUT);

  idft(result, result);

  normalize(result, result, 0, 1, NORM_MINMAX);
  split(result, planes);

  result = planes[0];
  result.convertTo(result, CV_8U, 255);

  std::cout << mean(original)[0] << " " << mean(result)[0] << std::endl;

  double diff = mean(original)[0] - mean(result)[0];
  result += Scalar::all(diff);
  cv::exp(result, result);

  Mat blurred;
  GaussianBlur(result, blurred, Size(3,3), 20);
  addWeighted(result, 1.3, blurred, -0.3, 0, result);

  std::cout << mse(result, original) << std::endl;

  imshow("result", result);
  imshow("input image", original);
  imshow("noised", noised);
  imshow("spectrum", mag);

  std::stringstream ss;
  ss << "lena_dft_filtered_mse=" << mse(result, original) << ".jpg";

  imwrite(ss.str(), result);
  waitKey();
}

void task3_6()
{
  original = imread(LENA_ORIGINAL);
  noised = imread(LENA_NOISED);

  bilateralFilter(noised, filtered, 9, 294, 4);
  medianBlur(filtered, filtered, 3);

  double MSError = mse(filtered, original);

  std::stringstream ss;
  ss << "lena_filtered_mse=" << MSError << ".jpg";
  imwrite(ss.str(), filtered);
}

int main()
{
  task3_5();
  return 0;
}

