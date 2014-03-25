#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


const std::string WALL_STR = "./images/wall/";
const std::string GRAF_STR = "./images/graf/";
const std::string BIKES_STR = "./images/bikes/";
const std::string LEUVEN_STR = "./images/leuven/";

const std::vector<std::string> images {"img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"};

enum Type {
  WALL,
  GRAF,
  BIKES,
  LEAUVEN
};

const std::map<Type, std::string> typeMap {
  {WALL, WALL_STR},
  {GRAF, GRAF_STR},
  {BIKES, BIKES_STR},
  {LEAUVEN, LEUVEN_STR}
};

typedef std::vector<std::vector<cv::KeyPoint>> Keypoints;

const std::string T0to1 = "images/T0to1";
const std::string T0to2 = "images/T0to2";
const std::string T0to3 = "images/T0to3";

const std::string transforms[] { T0to1, T0to2, T0to3 };

cv::Point applyTransform(const cv::Mat& transform, const cv::Point point, const cv::Mat& image);
cv::Mat readTransform(const std::string& path);

#endif // CONSTANTS_H
