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

#include <QtCore/QDirIterator>
#include <QtCore/QDir>
#include <QtCore/QDebug>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

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

  qDebug() << image_directory;

  cv::Mat descriptor_set;
  cv::Mat descriptors;

  //читаем не все картинки из-за нехватки памяти
  QStringList dirs = image_directory.entryList();
  qDebug() << dirs;

  for (int i = 0; i < dirs.size(); ++i)
  {
    if (dirs[i] == "." || dirs[i] == "..")
      continue;

    QDirIterator iterator(image_directory.absolutePath() + QDir::separator() + dirs[i]);
    std::cout << i << ": " << dirs[i].toStdString() << std::endl;

    int counter = 0;
    while (iterator.hasNext())
    {
      if (counter++ <= 20)
      {
        QString next = iterator.next();
        QFileInfo fileinfo = iterator.fileInfo();

        if (fileinfo.isFile())
        {
          std::cout << next.toStdString() << std::endl;
          std::vector<cv::KeyPoint> image_keypoints;
          std::vector<cv::Mat> image_patches;
          std::string image_path = next.toStdString();

          cv::Mat image = cv::imread(image_path);
          if (!image.empty())
          {
            detector.detect(image, image_keypoints);
            detector.compute(image, image_keypoints, descriptors);

            std::transform(image_keypoints.begin(), image_keypoints.end(), std::back_inserter(image_patches),
                          [&image](const cv::KeyPoint& keypoint){ return getKeypointPatch(keypoint, image); });

            descriptor_set.push_back(descriptors);
            keypoints.insert(keypoints.end(), image_keypoints.begin(), image_keypoints.end());
            keypoint_patches.insert(keypoint_patches.end(), image_patches.begin(), image_patches.end());
          }
        }
      }
      else
        break;
    }
  }

  std::cerr << "Building ClusterSpace..." << std::endl;
  ClusterSpace<double, 128> cluster_space(100);
  cluster_space.set_pictures(std::move(keypoint_patches));
  cluster_space.build(descriptor_set);
  std::cerr << "ClusterSpace has been built!" << std::endl;

  std::cout << "Size: " << cluster_space.size() << std::endl;
  size_t total = 0;
  std::for_each(cluster_space.begin(), cluster_space.end(), [&](const Cluster<double, 128>& cluster){
    total += cluster.size();
  });
  std::cout << "Total: " << total << std::endl;

  std::ofstream dump("cluster_space.txt");
  if (dump.is_open())
  {
    boost::archive::binary_oarchive oarchive(dump);
    oarchive << cluster_space;
    std::cout << "ClusterSpace dumped!" << std::endl;
  }
  dump.close();

  std::cerr << "Trying to read cluster space!" << std::endl;
  std::ifstream dedump("cluster_space.txt");
  ClusterSpace<double, 128> second_space;
  if (dedump.is_open())
  {
    boost::archive::binary_iarchive iarchive(dedump);
    iarchive >> second_space;
  }
  dedump.close();


  std::cout << "Size: " << second_space.size() << std::endl;
  total = 0;
  std::for_each(second_space.begin(), second_space.end(), [&](const Cluster<double, 128>& cluster){
    total += cluster.size();
  });
  std::cout << "Total: " << total << std::endl;


  std::ofstream compareDump("cluster_space_compare.txt");
  if (compareDump.is_open())
  {
    boost::archive::binary_oarchive oarchive(compareDump);
    oarchive << second_space;
    std::cout << "Comparing ClusterSpace dumped!" << std::endl;
  }
  compareDump.close();

//  std::vector<cv::Mat> patches = cluster_space.get_pictures_by_index(5);
//  int counter = 0;
//  for (const auto& image : patches)
//  {
//    std::stringstream ss;
//    ss << counter++ << ".jpg";
//    cv::imwrite(ss.str(), image);
//  }

  return 0;
}

