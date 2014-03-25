#include "siftmatcher.hpp"

#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

SiftMatcher::SiftMatcher(const Type type) : m_type(type)
{
  const std::string& imageDir = typeMap.find(type)->second;

  std::set<std::string> imageSet;
  for (auto it = images.begin(); it != images.end(); ++it)
      imageSet.insert(imageDir + *it);

  for (auto it = imageSet.begin(); it != imageSet.end(); ++it)
  {
    cv::Mat image = cv::imread(*it);
    std::cout << *it << std::endl;
    m_images.push_back(image);
  }
}

Keypoints SiftMatcher::detect()
{
  cv::SiftFeatureDetector detector;
  std::vector<std::vector<cv::KeyPoint>> keypoints;
  detector.detect(m_images, keypoints);
  return keypoints;
}

std::vector<cv::Mat> SiftMatcher::compute(Keypoints& keypoints)
{
  std::vector<cv::Mat> descriptors(keypoints.size());
  cv::SiftDescriptorExtractor extractor;

  for (size_t i = 0; i < keypoints.size(); ++i)
  {
    extractor.compute(m_images[i], keypoints[i], descriptors[i]);
  }

  return descriptors;
}

//const std::string T0to1 = "images/T0to1";
//const std::string T0to2 = "images/T0to2";
//const std::string T0to3 = "images/T0to3";

//const std::string transforms[] { T0to1, T0to2, T0to3 };

const std::string imagePaths[] {"img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"};

cv::Mat readTransform(const std::string& path)
{
  std::ifstream file(path);
  cv::Mat mat(3, 3, CV_32F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
  {
    float val; file >> val;
    mat.at<float>(i, j) = val;
  }

  return mat;
}

cv::Point applyTransform(const cv::Mat& transform, const cv::Point point, const cv::Mat& image)
{
  cv::Point result;

  int newX = point.x;
  int newY = point.y;

  result.x = transform.at<float>(0, 0) * newX +
             transform.at<float>(0, 1) * newY +
             transform.at<float>(0, 2);

  result.y = transform.at<float>(1, 0) * newX +
             transform.at<float>(1, 1) * newY +
             transform.at<float>(1, 2);

  result.y += transform.at<float>(0,1) * image.cols;

  return result;
}

void SiftMatcher::match()
{
  std::vector<std::vector<cv::KeyPoint>> keypoints = detect();
  std::vector<cv::Mat> descriptors = compute(keypoints);
  std::ofstream of(m_typeMap[m_type] + "SIFT/siftResults.txt");
  for (size_t i = 1; i < descriptors.size(); ++i)
  {

    const cv::Mat& desc1 = descriptors[0];
    const cv::Mat& desc2 = descriptors[i];
    cv::Mat transform = readTransform(transforms[i-1]);

    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < desc1.rows; ++i)
    {
      double dist = matches[i].distance;
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }

    std::cout << "Max dist: " << max_dist << std::endl;
    std::cout << "Min dist: " << min_dist << std::endl;

    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < matches.size(); ++i)
    {
      if (matches[i].distance < 2 * min_dist)
        goodMatches.push_back(matches[i]);
    }

    const int total = goodMatches.size();
    int correct = 0;

    for (cv::DMatch match : goodMatches) {
      cv::KeyPoint p1 = keypoints[0][match.trainIdx];
      cv::KeyPoint p2 = keypoints[i][match.queryIdx];

      cv::Point orig = p1.pt;
      cv::Point rotated = p2.pt;
      cv::Point transformed = applyTransform(transform, orig, m_images[0]);
      if (std::abs(transformed.x - rotated.x) <= 5 && std::abs(transformed.y - rotated.y) <= 5)
        ++correct;
    }

    cv::Mat output;
    cv::drawMatches(m_images[0], keypoints[0], m_images[i], keypoints[i], goodMatches, output);
    cv::imwrite(m_typeMap[m_type] +  "SIFT/" + imagePaths[0] + "->" + imagePaths[i] + ".jpg", output);
    of << imagePaths[0] << "->" << imagePaths[i] << ":\n"
                        << "Total matches: " << total << '\n'
                        << "Good matches: " << correct << '\n'
                        << "Persentage: " << (int)ceil((double)correct / (double)total)
                        << std::endl;
  }
}

void SiftMatcher::doTask()
{

}

const std::vector<cv::Mat>& SiftMatcher::getImages() const
{
  return m_images;
}
