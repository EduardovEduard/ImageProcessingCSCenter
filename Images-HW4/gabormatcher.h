#ifndef GABORMATCHER_H
#define GABORMATCHER_H

#include <opencv2/opencv.hpp>

#include <vector>
#include <utility>
#include <map>

typedef std::pair<std::pair<std::pair<std::string, cv::Mat*>, std::pair<std::string, cv::Mat*>>,double> ResultEntry;
typedef std::vector<ResultEntry>ResultBank; //lol

class GaborMatcher
{
public:
    void init();

    cv::Mat getKernel(int kernel_size, double sigma, double theta, double lambda, double psi);

    std::vector<std::pair<cv::Mat, cv::Mat>> applyAllFilters(const cv::Mat& image);

    std::vector<cv::Mat> getImages() const;

    void doTask();

    long double calcDistance(const cv::Mat& left, const cv::Mat& right) const;

    long double calcGaborDistance(const cv::Mat& left, const cv::Mat& right);

private:

    void generateHTML(const ResultBank& result);
    void printFile(const ResultBank& result);

    std::vector<double> calcFeatureVector(const cv::Mat& image);

    std::vector<cv::Mat> m_filters;
    std::map<std::string, cv::Mat> m_images;
    std::map<cv::Mat*, std::vector<double>> m_featureVectors;
};

#endif // GABORMATCHER_H
