#include "histogrammmatcher.h"

int main(int argc, char** argv)
{
  HistogrammMatcher matcher;
  matcher.init(18,18,18);
  matcher.doTask("TN_191005.JPG", HistogrammMatcher::CHI2);
  cv::waitKey(0);
  return 0;
}
