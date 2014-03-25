#ifndef HARRISMATCHER_HPP
#define HARRISMATCHER_HPP

#include "constants.h"

#include <opencv2/opencv.hpp>

#include <memory>

#include <string>
#include <vector>
#include <map>


class PointWrapper : public cv::Point
{
public:
  bool operator < (const PointWrapper& other) const
  {
    if (x != other.x) return x < other.x;
    else return y < other.y;
  }

  PointWrapper() : cv::Point() {}
  PointWrapper(int x, int y) : cv::Point(x,y) {}
};

typedef std::vector<double> Histogramm;
typedef std::map<PointWrapper, Histogramm> PointDescriptors;

struct DescriptorPair {
  PointWrapper p1;
  PointWrapper p2;
  double distance;
};

struct Match {
  std::shared_ptr<cv::Mat> im1;
  std::shared_ptr<cv::Mat> im2;
  std::vector<DescriptorPair> pairs;
};

class HarrisMatcher
{
public:

  HarrisMatcher(Type type);

  std::vector<Match> getSortedPairSet();

private:

  void detectCorners();

  void computeDescriptors();

  double histDistance(Histogramm&, Histogramm& h2);

  Histogramm computeDescriptor(const cv::Mat& image, const PointWrapper& point);

  std::shared_ptr<cv::Mat> m_originalImage;

  std::vector<std::shared_ptr<cv::Mat>> m_images;
  std::map<std::shared_ptr<cv::Mat>, std::vector<PointWrapper>> m_harrisCorners;
  std::map<std::shared_ptr<cv::Mat>, PointDescriptors> m_descriptors;
};

#endif // HARRISMATCHER_HPP
