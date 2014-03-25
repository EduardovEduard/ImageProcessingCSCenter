#include "knnfinder.h"
#include "clusterspace.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <string>
#include <list>
#include <random>

#include <QDirIterator>
#include <QDir>

const std::string PATH = "./data/mat-500-";
const std::string TXT = ".txt";

const std::string BOW_DIRECTORY = "101_ObjectCategories";

typedef std::vector<cv::Mat> ClassifiedImages;

//Прочитать все файлы с данными
cv::Mat read_data()
{
    cv::Mat data(DATA_SIZE, ROW_SIZE, CV_64FC1);

    std::vector<char> block;
    for (int i = 1; i <= 10; ++i)
    {
      std::stringstream ss;
      std::vector<double> entry;
      entry.resize(ROW_SIZE);

      ss << PATH << i << TXT;
      std::ifstream file(ss.str());

      if (file.is_open())
      {
        //Как можно быстрее читаем файл
        std::streambuf* buffer = file.rdbuf();

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        block.resize(size);
        buffer->sgetn(&block[0], size);

        std::string contents(block.begin(), block.end());
        std::istringstream stream(contents);

        //Разбираем файл по строкам
        int row = (i - 1) * (DATA_SIZE / 10);

        while (!stream.eof())
        {
          std::string line;
          std::getline(stream, line);

          //собираем данные из строки
          if (!line.empty())
          {
            double* row_ptr = data.ptr<double>(row++);
            std::istringstream line_stream(line);

            std::vector<std::string> values{ std::istream_iterator<std::string>(line_stream),
                                             std::istream_iterator<std::string>()};

            std::transform(values.begin(), values.end(), row_ptr, [&](const std::string& str){
              return std::stod(str);
            });
          }
        }
      }
      else
      {
          std::cerr << "Couldn't open " << ss.str() << std::endl;
      }
      std::cout << "File #" << i << " read successfully!" << std::endl;
    }

    std::cerr << "Data read successfully!" << std::endl;
    return std::move(data);
}

//Прочитать все картинки
ClassifiedImages read_categories()
{
  ClassifiedImages result;
  QDir image_directory(QString::fromStdString(BOW_DIRECTORY));
  QDirIterator dir_iterator(image_directory, QDirIterator::Subdirectories);

  while (dir_iterator.hasNext())
  {
    QString next = dir_iterator.next();
    QFileInfo fileinfo = dir_iterator.fileInfo();

    if (fileinfo.isFile())
    {
      std::string image_path = next.toStdString();
      cv::Mat image = cv::imread(image_path);
      result.push_back(image);
    }
  }

  return result;
}

//Вырезать из картинки кусок 16x16 по ключевой точке
cv::Mat getKeypointPatch(const cv::KeyPoint keypoint, const cv::Mat& image)
{
  cv::Point pt = keypoint.pt;
  float size = 16;
  float angle = keypoint.angle;

  cv::RotatedRect rect(pt, cv::Size2f(size, size), angle);

  cv::Mat m, rotated, cropped;
  cv::Size rect_size = rect.size;
  angle = rect.angle;

  if (rect.angle < -45.) {
    angle += 90.0;
    std::swap(rect_size.width, rect_size.height);
  }

  m = cv::getRotationMatrix2D(rect.center, angle, 1.0);
  cv::warpAffine(image, rotated, m, image.size(), cv::INTER_CUBIC);
  cv::getRectSubPix(rotated, rect_size, rect.center, cropped);

  return cropped;
}

int main()
{
// Эти 2 строчки делают 1 задание. Все остальное - 2е
//  KnnFinder finder(read_data());
//  finder.do_different_size_search();

  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Mat> keypoint_patches;

  cv::SiftFeatureDetector detector;

  ClassifiedImages images;
  QDir image_directory(QString::fromStdString(BOW_DIRECTORY));
  QDirIterator dir_iterator(image_directory, QDirIterator::Subdirectories);

  cv::Mat descriptor_set;
  cv::Mat descriptors;

  //читаем каждую третью картинку из-за нехватки памяти
  int counter = 0;
  int third = 0;
  while (dir_iterator.hasNext())
  {
    QString next = dir_iterator.next();
    QFileInfo fileinfo = dir_iterator.fileInfo();

    if (fileinfo.isFile())
    {
      std::vector<cv::KeyPoint> image_keypoints;
      std::vector<cv::Mat> image_patches;
      std::string image_path = next.toStdString();

      if (third == 0)
      {
        std::cout << counter++ << "'th image!" << std::endl;
        cv::Mat image = cv::imread(image_path);
        detector.detect(image, image_keypoints);
        detector.compute(image, image_keypoints, descriptors);

        std::transform(image_keypoints.begin(), image_keypoints.end(), std::back_inserter(image_patches),
                       [&image](const cv::KeyPoint& keypoint) { return getKeypointPatch(keypoint, image); });

        descriptor_set.push_back(descriptors);
        keypoints.insert(keypoints.end(), image_keypoints.begin(), image_keypoints.end());
        keypoint_patches.insert(keypoint_patches.end(), image_patches.begin(), image_patches.end());

        ++third;
      }
      else
      {
        ++third %= 3;
      }
    }
  }

  std::cerr << "Building ClusterSpace..." << std::endl;
  ClusterSpace<double, 128> cluster_space(100);
  cluster_space.set_pictures(std::move(keypoint_patches));
  cluster_space.build(descriptor_set);
  std::cerr << "ClusterSpace has been built!" << std::endl;

  std::ofstream dump("cluster_space.txt");
  if (dump.is_open())
  {
    dump << cluster_space;
    std::cout << "ClusterSpace dumped!" << std::endl;
  }

  std::vector<cv::Mat> patches = cluster_space.get_pictures_by_index(5);
  counter = 0;
  for (const auto& image : patches)
  {
    std::stringstream ss;
    ss << counter++ << ".jpg";
    cv::imwrite(ss.str(), image);
  }

  return 0;
}
