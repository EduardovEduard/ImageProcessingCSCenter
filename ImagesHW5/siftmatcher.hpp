#ifndef SIFTMATCHER_HPP
#define SIFTMATCHER_HPP

#include "constants.h"

#include <string>
#include <map>
#include <opencv2/opencv.hpp>

class SiftMatcher
{
public:
  SiftMatcher(const Type type);

  Keypoints detect();

  std::vector<cv::Mat> compute(Keypoints& keypoints);

  void match();

  void doTask();

  const std::vector<cv::Mat>& getImages() const;
private:

  Type m_type;

  std::vector<cv::Mat> m_images;

  std::map<Type, std::string> m_typeMap {
    {WALL, WALL_STR},
    {GRAF, GRAF_STR},
    {BIKES, BIKES_STR},
    {LEAUVEN, LEUVEN_STR}
  };
};

#endif // SIFTMATCHER_HPP
