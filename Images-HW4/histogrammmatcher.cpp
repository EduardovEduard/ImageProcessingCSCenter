#include "histogrammmatcher.h"

#include <QDir>
#include <QDirIterator>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>

void HistogrammMatcher::init(int a, int b, int c)
{
  QDir imageDir(QDir::currentPath() + QDir::separator() + "Corel/");
  QStringList imagePaths = imageDir.entryList(QDir::Files);

  for (QString path : imagePaths)
  {
    std::string str = path.toStdString();
    cv::Mat im = cv::imread("Corel/" + str);
    cv::cvtColor(im, im, CV_BGR2HSV);
    m_images[str] = im;
    m_hists[str] = calcHistogramm(im, a, b, c);
  }
}

double HistogrammMatcher::distance(const std::string &image1, const std::string &image2, Method method)
{
  auto histogramms1 = m_hists[image1];
  auto histogramms2 = m_hists[image2];

  double results[3];
  for (size_t i = 0; i < histogramms1.size(); ++i) {
    results[i] = method == L1 ? histogrammDiffL1(histogramms1[i], histogramms2[i])
                              : histogrammDiffChi2(histogramms1[i], histogramms2[i]);
  }
  std::cout << image1 << " : " << image2 << " -> " << results[0] << " " << results[1] << " " << results[2] << " -> " <<
               std::sqrt(std::pow(results[0], 2) + std::pow(results[1], 2) + std::pow(results[2], 2)) << std::endl;

  return std::sqrt(std::pow(results[0], 2) + std::pow(results[1], 2) + std::pow(results[2], 2));
//  return results[0] + results[1] / 3 + results[2] / 3;
}

void HistogrammMatcher::doTask(const std::string& target, HistogrammMatcher::Method method)
{
  std::vector<std::pair<std::string, double> > result;
  for (auto it = m_images.begin(); it != m_images.end(); ++it)
  {
    const std::string& test = it->first;
    if (test != target)
    {
      result.push_back(std::make_pair(test, distance(target, test, method)));
    }
  }

  std::sort(result.begin(), result.end(), [&](std::pair<std::string, double> a, std::pair<std::string, double> b) {
    return a.second < b.second;
  });

  std::ofstream ofs(method == L1 ? "L1.txt" : "ChiSq.txt");
  for (auto it = result.begin(); it != result.end(); ++it)
  {
    ofs << it->first << "\n";
  }

  for(int i = 0; i < 10; ++i)
  {
    std::stringstream ss;
    ss << result[i].second;
    cv::cvtColor(m_images[result[i].first], m_images[result[i].first], CV_HSV2BGR);
    cv::imshow(ss.str(), m_images[result[i].first]);
  }
  cv::cvtColor(m_images[target], m_images[target], CV_HSV2BGR);
  cv::imshow("TARGET", m_images[target]);

  ofs.flush();
  ofs.close();
}

Histogramms HistogrammMatcher::calcHistogramm(const cv::Mat &image, int a, int b, int c)
{
  const int max_value = 256;
  const int a_width = max_value / a;
  const int b_width = max_value / b;
  const int c_width = max_value / c;

  Histogramms result;
  result.push_back(Histogramm(a));
  result.push_back(Histogramm(b));
  result.push_back(Histogramm(c));

  for (auto it = image.begin<cv::Vec3b>(); it != image.end<cv::Vec3b>(); ++it)
  {
    ++result[0][std::ceil((*it)[0] / a_width)];
    ++result[1][std::ceil((*it)[1] / b_width)];
    ++result[2][std::ceil((*it)[2] / c_width)];
  }

  return result;
}

double HistogrammMatcher::histogrammDiffL1(Histogramm &h1, Histogramm &h2)
{
  assert(h1.size() == h2.size());
  normalize(h1);
  normalize(h2);

  double sum = 0;
  for (auto it1 = h1.begin(), it2 = h2.begin(); it1 != h1.end(); ++it1, ++it2)
    sum += std::min(*it1, *it2);

  return 1 - sum;
}

long double HistogrammMatcher::histogrammDiffChi2(Histogramm &h1, Histogramm &h2)
{
  assert(h1.size() == h2.size());

  double sum = 0;
   for (size_t i = 0; i < h1.size(); ++i)
  {
    if(h1[i] != 0 && h2[i] != 0)
      sum += std::pow(h1[i] - h2[i], 2) / (h1[i] + h2[i]);
  }

  return sum;
}

void HistogrammMatcher::normalize(Histogramm& hist)
{
  double summ = std::accumulate(hist.begin(), hist.end(), 0.0);
  for (size_t i = 0; i < hist.size(); ++i)
  {
    hist[i] /= summ;
  }
}
