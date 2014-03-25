#include "harrismatcher.hpp"

#include <boost/lexical_cast.hpp>

#include <fstream>
#include <set>

HarrisMatcher::HarrisMatcher(Type type)
{
  const std::string& ORIGINAL = images[0];

  std::set<std::string> imageSet;
  const std::string& imageDir = typeMap.find(type)->second;

  for (auto it = images.begin(); it != images.end(); ++it)
    imageSet.insert(imageDir + *it);

  for (auto it = imageSet.begin(); it != imageSet.end(); ++it)
  {
    std::cout << *it << std::endl;
    std::shared_ptr<cv::Mat> image = std::make_shared<cv::Mat>(cv::imread(*it));
    m_images.push_back(image);

    if (it->find(ORIGINAL) != std::string::npos)
      m_originalImage = image;
  }

  detectCorners();
}

std::vector<Match> HarrisMatcher::getSortedPairSet()
{
  std::vector<Match> matches;
  for (auto imageIt = m_images.begin(); imageIt != m_images.end(); ++imageIt)
  {
    if (*imageIt == m_originalImage)
      continue;

    Match match;
    match.im1 = m_originalImage;
    match.im2 = *imageIt;

    const PointDescriptors& originalDescriptors = m_descriptors[m_originalImage];
    const PointDescriptors& otherDescriptors = m_descriptors[*imageIt];
    std::cout << originalDescriptors.size() << " " << otherDescriptors.size() << std::endl;
    std::vector<DescriptorPair> result;

    double totalMinimumDistance = 1;
    for (auto otherPointHistPair : otherDescriptors)
    {
      DescriptorPair pair;
      pair.p2 = otherPointHistPair.first;
      double minDist = 1;

      for (auto origPointHistPair : originalDescriptors)
      {
        double distance = histDistance(otherPointHistPair.second, origPointHistPair.second);
        if (distance < minDist)
        {
          minDist = distance;
          pair.p1 = origPointHistPair.first;
          pair.distance = distance;
        }

        if (distance < totalMinimumDistance)
          totalMinimumDistance = distance;
      }

//      std::cout << pair.distance << std::endl;
      result.push_back(pair);
    }

//    std::vector<DescriptorPair> filteredResult;
//    for (auto it = result.begin(); it != result.end(); ++it)
//      if (it->distance < totalMinimumDistance * 2)
//        filteredResult.push_back(*it);

    std::sort(result.begin(), result.end(), [&](const DescriptorPair& p1, const DescriptorPair& p2) {
      return p1.distance > p2.distance;
    });

    match.pairs = result;
    matches.push_back(match);
  }

  return matches;
}

void HarrisMatcher::detectCorners()
{
  const int blockSize = 2;
  const int kSize = 3;
  const double k = 0.04;

  for (const std::shared_ptr<cv::Mat>& img : m_images)
  {
    const float HARRIS_THRESHOLD = 15;
    cv::Mat imageGray;
    cv::Mat harris = cv::Mat::zeros(img->size(), CV_32FC1);

    cv::cvtColor(*img, imageGray, CV_BGR2GRAY);

    cv::cornerHarris(imageGray, harris, blockSize, kSize, k);
    cv::normalize(harris, harris, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::convertScaleAbs(harris, harris);
    //Почему то, оригинал темнее чем повернутые картинки.
    //Подровняем всех вычитая цвет фона
    harris -= (harris.at<uchar>(0,0));

    for (int i = 0; i < harris.cols; ++i)
      for (int j = 0; j < harris.rows; ++j)
        if (harris.at<uchar>(j,i) > HARRIS_THRESHOLD)
          m_harrisCorners[img].push_back(PointWrapper(i,j));
  }
  computeDescriptors();
}

void HarrisMatcher::computeDescriptors()
{
  for (auto imageIt = m_harrisCorners.begin(); imageIt != m_harrisCorners.end(); ++imageIt)
  {
    PointDescriptors descriptors;

    cv::Mat hsvImage;
    cv::cvtColor(*imageIt->first, hsvImage, CV_BGR2HSV);

    for (const PointWrapper& keypoint : imageIt->second)
    {
      descriptors[keypoint] = computeDescriptor(hsvImage, keypoint);
    }

    m_descriptors[imageIt->first] = descriptors;
  }
}

void normalize(Histogramm& hist)
{
  double summ = std::accumulate(hist.begin(), hist.end(), 0.0);
  for (size_t i = 0; i < hist.size(); ++i)
  {
    hist[i] /= summ;
  }
}

double HarrisMatcher::histDistance(Histogramm& h1, Histogramm& h2)
{
  assert(h1.size() == h2.size());
  normalize(h1);
  normalize(h2);

//  std::cout << std::endl;
//  for (double i : h1)
//    std::cout << i << " ";
//  std::cout << std::endl;

  double sum = 0;
   for (size_t i = 0; i < h1.size(); ++i)
  {
    if(h1[i] != 0 && h2[i] != 0)
      sum += std::pow(h1[i] - h2[i], 2) / (h1[i] + h2[i]);
  }

  return sum;
}

Histogramm HarrisMatcher::computeDescriptor(const cv::Mat& image, const PointWrapper& point)
{
  const int DESCRIPTOR_SIZE = 20;
  const int X = point.x, Y = point.y;
  const int MAX_VALUE = 256;
  const int BUCKET_COUNT = 18;
  const int BUCKET_WIDTH = MAX_VALUE / BUCKET_COUNT;

  Histogramm hist(BUCKET_COUNT);
  std::fill(hist.begin(), hist.end(), 0);

  for (int i = -DESCRIPTOR_SIZE; i < DESCRIPTOR_SIZE - 1; ++i)
  {
    for (int j = -DESCRIPTOR_SIZE; j < DESCRIPTOR_SIZE - 1; ++j)
    {
      if (X + i < 0 || X + i > image.cols)
        continue;

      if (Y + i < 0 || Y + i > image.rows)
        continue;

      ++hist[std::ceil(image.at<cv::Vec3b>(Y + i, X + i)[0] / BUCKET_WIDTH)];
    }
  }

//  std::cout << std::endl;
//  for (int i : hist)
//    std::cout << i << " ";
//  std::cout << std::endl;


  return hist;
}
