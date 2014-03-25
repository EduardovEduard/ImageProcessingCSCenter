#ifndef SHAPEMATCHER_H
#define SHAPEMATCHER_H

#include <opencv2/opencv.hpp>

#include <map>
#include <vector>
#include <string>

typedef std::vector<cv::Point> Contour;
typedef std::vector<Contour> Contours;
typedef std::vector<double> Signature;
typedef std::vector<double> FourierSignature;
typedef std::pair<std::pair<std::pair<std::string, cv::Mat*>, std::pair<std::string, cv::Mat*>>, double> MatchPair; //lol
typedef std::vector<MatchPair> ResultBank;

class ShapeMatcher
{
public:
  void init();
  void doTask();
private:
  std::map<std::string, cv::Mat> m_images;
  std::map<std::string, Contour> m_contours;
  std::map<std::string, Signature> m_signatures;
  std::map<std::string, FourierSignature> m_fourierSigs;

  Contour getContour(const cv::Mat& image);
  Contour sample(const Contour& sig, int rate);
  Signature getSignature(const Contour& contour);
  FourierSignature getFourierSignature(const Signature& signature);
  double diff(const FourierSignature& left, const FourierSignature& right);

  void generateHTML(const ResultBank& result);
  void printFile(const ResultBank& result);
};

#endif // SHAPEMATCHER_H
