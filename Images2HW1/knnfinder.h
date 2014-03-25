#ifndef KNNFINDER_H
#define KNNFINDER_H

#include <vector>
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/linear_index.h>
#include <memory>

const int DATA_SIZE = 1000000;
const int ROW_SIZE = 500;

using namespace cvflann;

class KnnFinder
{
public:
    KnnFinder(cv::Mat&& data);

    void do_different_dimensions_search();

    void do_different_size_search();

private:
    cv::Mat m_data;

    const std::vector<std::shared_ptr<cvflann::IndexParams>> params = {
            std::make_shared<LinearIndexParams>(),
            std::make_shared<KDTreeIndexParams>(),
            std::make_shared<KMeansIndexParams>(),
            std::make_shared<CompositeIndexParams>()
    };

    std::vector<std::string> param_names = { "LinearIndex", "KDTreeIndex", "KMeansIndex", "CompositeIndex", "LshIndex" };

};

#endif // KNNFINDER_H
