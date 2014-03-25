#include "gabormatcher.h"

#include <QDirIterator>
#include <QDir>

#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>

#include <fstream>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

void GaborMatcher::init()
{
    int count = 0;
    for (int sigmas = 2; sigmas <= 10; sigmas += 2)
    {
        for (int thetas = 0; thetas < 157; thetas += 22.5)
        {
          cv::Mat filter = getKernel(21, sigmas, thetas, 0.5, 90);
          m_filters.push_back(filter);
        }
    }

    std::cout << count << std::endl;
    QDir imagesDir(QDir::currentPath() + QDir::separator() + "brodatz");
    QStringList imageList = imagesDir.entryList(QDir::Files);

    for (QString path : imageList)
    {
        std::string str = path.toStdString();
        cv::Mat im = cv::imread("brodatz/" + str, CV_LOAD_IMAGE_GRAYSCALE);
        im.convertTo(im, CV_64F, 1.0 / 255);
        m_images[str] = im;
    }
}

cv::Mat GaborMatcher::getKernel(int kernel_size, double sig, double th, double lambda, double ps)
{
    int hks = (kernel_size - 1) / 2;

    double theta = th * CV_PI / 180;
    double psi = ps * CV_PI / 180;
    double del = 2.0 / (kernel_size - 1);
    double lmbd = lambda;
    double sigma = sig / kernel_size;
    double x_theta;
    double y_theta;

    cv::Mat kernel(kernel_size,kernel_size, CV_64F);

    for (int y = -hks; y <= hks; y++)
    {
        for (int x = -hks; x <= hks; x++)
        {
            x_theta = x * del * cos(theta) + y * del * sin(theta);
            y_theta = -x * del * sin(theta) + y * del * cos(theta);
            kernel.at<double>(hks + y, hks + x) = (double)exp(-0.5 * (pow(x_theta, 2) + pow(y_theta, 2)) / pow(sigma, 2)) * cos(2 * CV_PI * x_theta / lmbd + psi);
        }
    }
    return kernel;
}

std::vector<std::pair<cv::Mat, cv::Mat >> GaborMatcher::applyAllFilters(const cv::Mat& image)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> result;
    for (const cv::Mat& filter : m_filters)
    {
        cv::Mat filtered, mag;
        cv::filter2D(image, filtered, CV_32F, filter);
        cv::pow(filtered, 2, mag);
        result.push_back(std::make_pair(filtered, mag));
    }
    return result;
}

std::vector<cv::Mat> GaborMatcher::getImages() const
{
  std::vector<cv::Mat> result;
  for (auto it = m_images.begin(); it != m_images.end(); ++it)
  {
    result.push_back(it->second);
  }
  return result;
}

typedef std::vector<double> FeatureVector;
double calcFeatureDistance(const FeatureVector& left, const FeatureVector& right)
{
  assert(left.size() == right.size());

  double sum = 0;
  for (size_t i = 0; i < left.size(); ++i)
  {
      sum += std::pow(left[i] - right[i], 2);
  }
  return std::sqrt(sum);
}

void GaborMatcher::doTask()
{
  typedef std::vector<std::pair<std::pair<std::string, cv::Mat*>, FeatureVector>> PairVector;
  PairVector vector;

  ResultBank distanceVector;

  for (auto it = m_images.begin(); it != m_images.end(); ++it)
  {
      vector.push_back(std::make_pair(std::make_pair(it->first, &it->second), calcFeatureVector(it->second)));
  }

  for (PairVector::iterator it = vector.begin(); it != vector.end(); ++it)
  {
    for (PairVector::iterator it2 = it + 1; it2 != vector.end(); ++it2)
    {
       distanceVector.push_back(std::make_pair(std::make_pair(std::make_pair(it->first.first, it->first.second),
                                                              std::make_pair(it2->first.first, it2->first.second)),
                                               calcFeatureDistance(it->second, it2->second)));
    }
  }

  std::sort(distanceVector.begin(), distanceVector.end(),
            [&](const ResultEntry& a, const ResultEntry& b){
        return a.second < b.second;
    });

  generateHTML(distanceVector);
  printFile(distanceVector);
}

long double GaborMatcher::calcGaborDistance(const cv::Mat &left, const cv::Mat &right)
{
  typedef std::pair<cv::Mat, cv::Mat > ImagePair;

  std::vector<ImagePair> leftResult = applyAllFilters(left);
  std::vector<ImagePair> rightResult = applyAllFilters(right);
  std::vector<long double> distances;
  std::transform(leftResult.begin(), leftResult.end(), rightResult.begin(), std::back_inserter(distances),
                 [&](const ImagePair& leftPair, const ImagePair& rightPair) -> long double {
                   return calcDistance(leftPair.first, rightPair.first);
                 });

  return std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
}

void GaborMatcher::generateHTML(const ResultBank& result)
{
  std::ofstream html("gabor.html");

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
    fmt % (it->first.first.first);
    fmt % (it->first.second.first);
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
  html.close();
}

void GaborMatcher::printFile(const ResultBank& result)
{
  std::ofstream ofs("gaborOutput.txt");
  for (auto it = result.begin(); it != result.end(); ++it)
  {
    ofs << it->first.first.first << " " << it->first.second.first << " -> " << it->second << "\n";
  }
  ofs.flush();
  ofs.close();
}

std::vector<double> GaborMatcher::calcFeatureVector(const cv::Mat &image)
{
  std::vector<double> result;
  std::vector<std::pair<cv::Mat, cv::Mat >> filterResult = applyAllFilters(image);

  for (auto it = filterResult.begin(); it != filterResult.end(); ++it)
  {
    double energy = std::accumulate(it->first.begin<double>(), it->first.end<double>(), 0.0);
    result.push_back(energy);
  }

  return result;
}

long double GaborMatcher::calcDistance(const cv::Mat &left, const cv::Mat &right) const
{
  assert(left.size == right.size);

  std::vector<long double> distances;

  const int SIZE = 5;
  for (int i = 0; i < left.rows / SIZE; i++)
  {
    for (int j = 0; j < left.cols / SIZE; j++)
    {
      cv::Rect rect(j * SIZE, i * SIZE, SIZE, SIZE);
      cv::Mat leftRoi(left, rect), rightRoi(right, rect);

      long double sum = 0;
      for (int y = 0; y < leftRoi.rows; ++y)
      {
        for (int x = 0; x < leftRoi.cols; ++x)
        {
            //sum += fabs((leftRoi.at<double>(y,x) - rightRoi.at<double>(y,x)));
        }
      }
      distances.push_back(sum);// / (leftRoi.rows + leftRoi.cols);
    }
  }

  long double result = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
  return result;
}
