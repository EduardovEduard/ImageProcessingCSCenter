#ifndef HISTOGRAMMMATCHER_H
#define HISTOGRAMMMATCHER_H

#include <opencv2/opencv.hpp>

#include <map>
#include <string>
#include <array>
#include <inttypes.h>

typedef std::vector<double> Histogramm;
typedef std::vector<Histogramm> Histogramms;

class HistogrammMatcher
{
public:

  enum Method {
    L1,
    CHI2
  };

  void init(int a, int b, int c);

  double distance(const std::string& image1, const std::string& image2, Method method);

  void doTask(const std::string &target, Method method);

private:
  std::map<std::string, cv::Mat> m_images;
  std::map<std::string, Histogramms> m_hists;

  Histogramms calcHistogramm(const cv::Mat& image, int a, int b, int c);
  
  double histogrammDiffL1(Histogramm &h1, Histogramm &h2);
  
  long double histogrammDiffChi2(Histogramm& h1, Histogramm& h2);

  void normalize(Histogramm& hist);
};

#endif // HISTOGRAMMMATCHER_H
