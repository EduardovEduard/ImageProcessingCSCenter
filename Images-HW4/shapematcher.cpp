#include "shapematcher.h"

#include <iostream>
#include <functional>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <fstream>

#include <QDir>
#include <QStringList>
#include <QDebug>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

const std::string IMAGES_DIR = "leaves/";

void ShapeMatcher::init()
{
  QDir imagesDir(QString::fromStdString(IMAGES_DIR));
  const QStringList& images = imagesDir.entryList(QDir::Files);
  for (QString path : images)
  {
    const std::string& stdPath = path.toStdString();
    cv::Mat image = cv::imread(IMAGES_DIR + stdPath, CV_LOAD_IMAGE_GRAYSCALE);
    m_contours[stdPath] = sample(getContour(image), 512);
    m_signatures[stdPath] = getSignature(m_contours[stdPath]);
    m_fourierSigs[stdPath] = getFourierSignature(m_signatures[stdPath]);
    m_images[stdPath] = image;
  }
}

void ShapeMatcher::doTask()
{
  ResultBank result;
  for (auto it1 = m_fourierSigs.begin(); it1 != m_fourierSigs.end(); ++it1)
    for (auto it2 = it1; it2 != m_fourierSigs.end(); ++it2)
    {
      if (it1 != it2)
      {
        const std::string& leftPath = it1->first;
        const std::string& rightPath = it2->first;
        cv::Mat* left = &m_images[leftPath];
        cv::Mat* right = &m_images[rightPath];
        result.push_back(std::make_pair(std::make_pair(std::make_pair(leftPath, left), std::make_pair(rightPath, right)),
        diff(m_fourierSigs[it1->first], m_fourierSigs[it2->first])));
      }
    }

  std::sort(result.begin(), result.end(), [](const MatchPair& p1, const MatchPair& p2) {
    return p1.second < p2.second;
  });

  generateHTML(result);
  printFile(result);
}

Contour ShapeMatcher::getContour(const cv::Mat& image)
{
  cv::Mat mat(image.size(), image.type());
  cv::threshold(image, mat, 100, 255, CV_THRESH_BINARY_INV);
  cv::Mat str = cv::Mat::ones(3, 3, image.type());
  cv::morphologyEx(mat, mat, CV_MOP_CLOSE, str);
  cv::copyMakeBorder(mat, mat, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

  Contours contours;
  cv::findContours(mat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  assert(contours.size() == 1);
  return contours[0];
}

Signature ShapeMatcher::getSignature(const Contour& contour)
{
  //Using centroid distance signature
  Signature result(contour.size());
  double centerX, centerY;

  //find centroid
  for (const cv::Point& point : contour)
  {
    centerX += point.x; centerY += point.y;
  }
  centerX /= contour.size(); centerY /= contour.size();

  //calculate signature
  for (size_t i = 0; i < contour.size(); ++i)
  {
    result[i] = std::sqrt(std::pow(contour[i].x, 2) + std::pow(contour[i].y, 2));
  }
  return result;
}

FourierSignature ShapeMatcher::getFourierSignature(const Signature& signature)
{
  cv::Mat wrapper(signature), complex;
  cv::Mat planes[] = {cv::Mat_<double>(wrapper), cv::Mat::zeros(wrapper.size(), CV_64F)};

  cv::merge(planes, 2, complex);

  cv::dft(complex, complex);

  cv::Mat dftResult[2];
  cv::split(complex, dftResult);

  FourierSignature dftSignature(dftResult[0].begin<double>(), dftResult[0].end<double>());
  dftSignature.resize(dftSignature.size() / 2);

  FourierSignature result(dftSignature.size());

  std::transform(dftSignature.begin(), dftSignature.end(), result.begin(), [&](double val) {
    return val / dftSignature[0];
  });

  result.erase(result.begin());
  return result;
}

double ShapeMatcher::diff(const FourierSignature& left, const FourierSignature& right)
{
  assert(left.size() == right.size());

  double sum = 0;
  for (int i = 0; i < left.size(); ++i)
  {
    sum += std::pow(left[i]  - right[i] ,2);
  }
  return std::sqrt(sum);
}

void ShapeMatcher::generateHTML(const ResultBank& result)
{
  std::ofstream html("ShapeHtml.html");

  std::string htmlTag = "<html>%s</html>";
  std::string headTag = "<head><link rel=\"stylesheet\" href=\"style.css\" /></head>";
  std::string bodyTag = "<body>%s</body>";
  std::string mainBlock = "<div id = \"main\">%s</div>>";
  std::string imageBlock = "<div id = \"block\">\n"
                           "<div class = \"picture\"><img src = \"%s\"/></div>"
                           "<div class = \"picture\"><img src = \"%s\"/></div>"
                           "<div class = \"number\">%s</div></div>";

  std::string images;
  for (auto it = result.begin(); it != result.end(); ++it)
  {
    std::string currentImageBlock = imageBlock;
    boost::format fmt(currentImageBlock);
    fmt % (it->first.first.first + ".jpg");
    fmt % (it->first.second.first + ".jpg");
    fmt % boost::lexical_cast<std::string>(it->second);
    images += fmt.str();
  }

  boost::format mainBlockFormat(mainBlock);
  boost::format bodyFormat(bodyTag);
  boost::format htmlFormat(htmlTag);

  mainBlockFormat % images;
  bodyFormat % mainBlockFormat.str();
  std::string headWithAll = headTag + bodyFormat.str();

  htmlFormat % headWithAll;
  std::string resultHTML = htmlFormat.str();
  html << resultHTML;

  html.flush();
  html.close();
}

void ShapeMatcher::printFile(const ResultBank& result)
{
  std::ofstream ofs("shapeOutput.txt");
  for (auto it = result.begin(); it != result.end(); ++it)
  {
    ofs << it->first.first.first << " " << it->first.second.first << " -> " << it->second << "\n";
  }

  ofs.flush();
  ofs.close();
}

Contour ShapeMatcher::sample(const Contour& cont, int rate)
{
  Contour result;
  double take = (double)cont.size() / rate, step = 0;
  for (int i = 0; i < cont.size(); i = std::round(step + take), step += take)
  {
    result.push_back(cont[i]);
  }
  return result;
}
