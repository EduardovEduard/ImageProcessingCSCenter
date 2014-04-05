#ifndef CLUSTERSPACE_H
#define CLUSTERSPACE_H

#include "Cluster.h"

#include <vector>
#include <map>
#include <algorithm>
#include <ostream>
#include <istream>

#include <opencv2/opencv.hpp>
#include <boost/serialization/serialization.hpp>

template <class T, size_t Dim>
class ClusterSpace
{
public:
  typedef typename Cluster<T, Dim>::Descriptor Descriptor;
  typedef typename Cluster<T, Dim>::LabeledDescriptor LabeledDescriptor;
  typedef typename Cluster<T, Dim>::DataStorage DataStorage;

  ClusterSpace() : m_k(0), m_currentIndex(0)
  {
  }

  ClusterSpace(int k) : m_k(k), m_currentIndex(0)
  {
  }

  void build(const cv::Mat& data)
  {
    std::vector<int> best_labels;
    cv::Mat centers(m_k, data.cols, CV_64F);

    std::cerr << "Starting KMeans..." << std::endl;
    cv::kmeans(data, m_k, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0 ),
               1 ,cv::KMEANS_PP_CENTERS, centers);

    assert(best_labels.size() == data.rows);
    assert(centers.rows == m_k);
    assert(centers.cols == data.cols);

    std::cerr << "KMeans finished!" << std::endl;
    std::cerr << "Filling internal structures!" << std::endl;

    for (int i = 0; i < m_k; ++i)
    {
      T* row_pointer = centers.ptr<T>(i);
      m_clusters.emplace_back(i, std::vector<T>(row_pointer, row_pointer + centers.cols));
    }

    for (size_t i = 0; i < best_labels.size(); ++i)
    {
      const T* row_ptr = data.ptr<T>(i);
      Descriptor descriptor;
      std::copy(row_ptr, row_ptr + data.cols, descriptor.begin());
      m_clusters[best_labels[i]].add_data(m_currentIndex++, descriptor);
    }
  }

  void set_pictures(std::vector<cv::Mat>&& pictures)
  {
    m_pictures = std::move(pictures);
    m_maxIndex = pictures.size();
  }

  typename std::vector<Cluster<T, Dim>>::const_iterator begin() const
  {
    return m_clusters.begin();
  }

  typename std::vector<Cluster<T, Dim>>::const_iterator end() const
  {
    return m_clusters.end();
  }

  size_t size() const
  {
    return m_clusters.size();
  }

  std::vector<cv::Mat> get_pictures_by_index(size_t index) const
  {
    std::vector<cv::Mat> result;
    std::vector<size_t> indexes = m_clusters[index].indexes();
    for (size_t i : indexes)
    {
      result.push_back(m_pictures[i]);
    }
    return result;
  }

  template <class Archive>
  void serialize(Archive& archive, const int)
  {
    archive & m_k;
    archive & m_clusters;
  }

private:
  std::vector<Cluster<T, Dim>> m_clusters;
  std::vector<cv::Mat> m_pictures;

  int m_k;
  int m_maxIndex;
  int m_currentIndex;

};


#endif // CLUSTERSPACE_H
