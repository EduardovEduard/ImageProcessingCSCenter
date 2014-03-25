#ifndef BLOBDETECTOR_HPP
#define BLOBDETECTOR_HPP

#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <sys/syscall.h>

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

const std::string IMAGES_DIR = "images/";
const int SCALE_COUNT = 3;
const double STEP = std::sqrt(std::sqrt(2.0));
const double SIGMA = 1.6;
const double SCALE_SPACE_STEP = 0.4;

typedef std::vector<std::pair<cv::Point, double> > BlobCenters;
typedef std::vector<std::pair<cv::Mat, double> > ScaleSpace;


class LoG
{
public:
  static constexpr double THRESHOLD = 0.35;

  static ScaleSpace createScaleSpace(const cv::Mat& image)
  {
    ScaleSpace space;
    assert(image.type() == CV_32F);

    double t = 0.5;
    while (t <= 3.5)
    {
      const double sigma = exp(t);
      const int kernelSize = ceil(3 * sigma) * 2 + 1;

      cv::Mat resultImage(image.size(), image.type());
      cv::GaussianBlur(image, resultImage, cv::Size(kernelSize, kernelSize), sigma);
      cv::Laplacian(resultImage, resultImage, CV_32F);

      resultImage *= std::pow(sigma, 2);
      resultImage = cv::abs(resultImage);

      space.push_back(std::make_pair(resultImage ,sigma));
      t += SCALE_SPACE_STEP;
    }

//    for (auto it = space.begin(); it != space.end(); ++it)
//    {
//      cv::imshow(boost::lexical_cast<std::string>(it->second), it->first);
//    }
//    cv::waitKey();

    return space;
  }
};

class DoG
{
public:
  static constexpr double THRESHOLD = 0.7;

  static ScaleSpace createScaleSpace(const cv::Mat& image)
  {
    ScaleSpace space;

    double t = 1.0;
    while (t <= 3.2)
    {
      const double sigma = exp(t);
      const int size = ceil(3 * sigma) * 2 + 1;

      cv::Mat resultImage;
      cv::GaussianBlur(image, resultImage, cv::Size(size, size), sigma);
      space.push_back(std::make_pair(resultImage, sigma));

      t += SCALE_SPACE_STEP;
    }

    for (size_t i = 0; i < space.size() - 1; ++i)
    {
      space[i].first = space[i].first - space[i + 1].first;
      space[i].first *= std::pow(space[i].second, 2);
      space[i].first = cv::abs(space[i].first);
      cv::normalize(space[i].first, space[i].first, 0, 1.0, cv::NORM_MINMAX ,CV_32F);
    }

    for (auto it = space.begin(); it != space.end(); ++it)
    {
      cv::imshow(boost::lexical_cast<std::string>(it->second), it->first);
    }
//    cv::waitKey();
//    exit(0);

    return space;
  }
};

template<class Strategy>
class BlobDetector
{
public:
  BlobDetector(const cv::Mat& image)
  {
    m_originalImage = image;
    cv::cvtColor(m_originalImage, m_grayImage, CV_BGR2GRAY);
    m_grayImage.convertTo(m_grayImage, CV_32F, 1.0 / 255);

    m_scaleSpace = Strategy::createScaleSpace(m_grayImage);
  }

  cv::Mat highlightBlobs()
  {
    BlobCenters centers = detectBlobs();
    cv::Mat image = m_originalImage;

    for (int i = 0; i < centers.size(); ++i)
    {
      cv::circle(image, centers[i].first,
                 centers[i].second * std::sqrt(2), cv::Scalar(0,0,255), 2);
    }

    return image;
  }

private:
  std::vector<float> getNeighbourValues(cv::Point point, ScaleSpace space, int spaceLoc)
  {
    std::vector<float> result;
    const cv::Mat& currentSpace = space[spaceLoc].first;

    cv::Mat highSpace, lowSpace;

    if (spaceLoc + 1 < space.size())
      highSpace = space[spaceLoc + 1].first;

    if (spaceLoc - 1 >= 0)
      lowSpace = space[spaceLoc - 1].first;

    int x = point.x;
    int y = point.y;

    for (int i = -1; i <= 1; ++i)
    {
      for (int j = -1; j <= 1; ++j)
      {
        if (x + i < 0 || x + i > currentSpace.cols - 1)
          continue;

        if (y + j < 0 || y + j > currentSpace.rows - 1)
          continue;

        if (i != 0 || j != 0)
          result.push_back(currentSpace.at<float>(y + j, x + i));

        if (!highSpace.empty())
          result.push_back(highSpace.at<float>(y + j, x + i));

        if (!lowSpace.empty())
          result.push_back(lowSpace.at<float>(y + j, x + i));
      }
    }

    assert(result.size() == 26 || result.size() == 17 || result.size() == 11);
    return result;
  }

  BlobCenters detectBlobs()
  {
    assert(m_scaleSpace.size() > 2);

    BlobCenters result;
    const int size = m_scaleSpace.size();

    for (int i = 1; i < size - 1; ++i)
    {
      const cv::Mat& currentScale = m_scaleSpace[i].first;
      double currentSig = m_scaleSpace[i].second;
      for (int x = 0; x < currentScale.cols; ++x)
      {
        for (int y = 0; y < currentScale.rows; ++y)
        {
          std::vector<float> neighbours = getNeighbourValues(cv::Point(x,y), m_scaleSpace, i);
          std::sort(neighbours.begin(), neighbours.end());
          std::reverse(neighbours.begin(), neighbours.end());

  //        std::cout << neighbours.size() << std::endl;
          if (currentScale.at<float>(y,x) > neighbours[0] && currentScale.at<float>(y,x) > Strategy::THRESHOLD) //local maximum
          {
            result.push_back(std::make_pair(cv::Point(x,y), currentSig));
          }
        }
      }
    }

    std::cout << result.size() << std::endl;

    m_blobs = result;
    return result;
  }

  ScaleSpace m_scaleSpace;
  BlobCenters m_blobs;
  cv::Mat m_originalImage;
  cv::Mat m_grayImage;
};

#endif // BLOBDETECTOR_HPP
