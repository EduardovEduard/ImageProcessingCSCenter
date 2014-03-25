#include "blobdetector.hpp"
#include "siftmatcher.hpp"
#include "harrismatcher.hpp"

#include <fstream>

int main()
{
    cv::Mat image = cv::imread("/home/ees/Pictures/images.jpeg");
  BlobDetector<LoG> detector(image);
  cv::Mat blobs = detector.highlightBlobs();
  cv::imshow("b", blobs);
  cv::waitKey();
  return 0;

  const std::string WALL_STR = "./images/wall/";
  const std::string GRAF_STR = "./images/graf/";
  const std::string BIKES_STR = "./images/bikes/";
  const std::string LEUVEN_STR = "./images/leuven/";

  const std::string stringArr[] {WALL_STR, GRAF_STR, BIKES_STR, LEUVEN_STR};

  for (int imagePack = 0; imagePack < 4; ++imagePack)
  {
    HarrisMatcher matcher((Type)imagePack);
    std::vector<Match> result = matcher.getSortedPairSet();
    std::ofstream resultFile(stringArr[imagePack] + "Harris/" + "results.txt");

    int i = 0;
    for (const Match& match : result)
    {
      cv::Mat im1 = match.im1->clone();
      cv::Mat im2 = match.im2->clone();

      cv::Mat stitched = cv::Mat::zeros(std::max(im1.rows, im2.rows), im1.cols + im2.cols, CV_8UC3);
      cv::Mat left(stitched, cv::Rect(0, 0, im1.cols, im1.rows));
      cv::Mat right(stitched, cv::Rect(im1.cols, 0, im2.cols, im2.rows));

      cv::Mat transform = readTransform(transforms[i]);

      im1.copyTo(left);
      im2.copyTo(right);

      cv::RNG rng;
      int totalPairs = match.pairs.size();
      int correctCount = 0;

      for (auto it = match.pairs.begin(); it != match.pairs.end(); ++it)
      {
        cv::Point orig = it->p2;
        cv::Point transformed = applyTransform(transform, it->p1, im1);

        bool isCorrect = false;
        if (std::abs(orig.x - transformed.x) <= 5 && std::abs(orig.y - transformed.y) <= 5)
        {
          ++correctCount;
          isCorrect = true;
        }

        /*if (it->distance < 0.1)*/ {
          cv::Point p(it->p2.x + im1.cols, it->p2.y);
          cv::Scalar color = isCorrect ? cv::Scalar(0,0,0) : cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
          cv::circle(stitched, it->p1, 3, color, 3);
          cv::circle(stitched, p, 3, color, 3);
          cv::line(stitched, it->p1, p, color);
        }
      }

      static std::string pics[] { "img1", "img2", "img3"};
      cv::imwrite(stringArr[imagePack] + "Harris/" + std::string("img0->") + pics[i] + ".jpg", stitched);
      resultFile << "img0 -> " << pics[i] << ":\n"
                 << "Total pairs: " << totalPairs << '\n'
                 << "Correct pairs: " << correctCount << '\n'
                 << "Persentage: " << (int)ceil(((double)correctCount / (double)totalPairs) * 100) << "%\n" << std::endl;
      ++i;
    }
    resultFile.close();
  }

  for (int i = 0; i < 4; ++i)
  {
    SiftMatcher matcher((Type)i);
    matcher.match();
  }

  return 0;
}
