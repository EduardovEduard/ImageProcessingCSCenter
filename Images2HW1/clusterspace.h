#ifndef CLUSTERSPACE_H
#define CLUSTERSPACE_H

#include "Cluster.h"

#include <vector>
#include <map>
#include <algorithm>
#include <ostream>
#include <istream>

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

template <class T, size_t Dim>
class ClusterSpace
{
public:
  typedef typename Cluster<T, Dim>::Descriptor Descriptor;
  typedef typename Cluster<T, Dim>::LabeledDescriptor LabeledDescriptor;
  typedef typename Cluster<T, Dim>::DataStorage DataStorage;

  ClusterSpace(int k) : m_k(k), m_currentIndex(0)
  {
  }

  void build(const cv::Mat& data)
  {
    std::vector<int> best_labels;
    std::vector<std::vector<T>> centers;

    std::cerr << "Starting KMeans..." << std::endl;
    cv::kmeans(data, m_k, best_labels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0 ),
               1 ,cv::KMEANS_PP_CENTERS, centers);

    assert(best_labels.size() == data.rows);
    assert(centers.size() == data.rows);
    assert(centers[0].size() == data.cols);

    std::cerr << "KMeans finished!" << std::endl;
    std::cerr << "Filling internal structures!" << std::endl;

    for (int i = 0; i < m_k; ++i)
    {
      m_clusters[i] = Cluster<T, Dim>(i, centers[i]);
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

private:
  std::vector<Cluster<T, Dim>> m_clusters;
  std::vector<cv::Mat> m_pictures;

  int m_k;
  int m_maxIndex;
  int m_currentIndex;

  friend std::ostream& operator<<(std::ostream& stream, const ClusterSpace<T, Dim>& cluster_space)
  {
    std::cerr << "ClusterSpace serialization started!" << std::endl;

    stream << cluster_space.size() << std::endl;
    for (const auto& cluster_it : cluster_space)
    {
      const Cluster<T, Dim>& cluster = cluster_it;
      const int label = cluster.label();
      const size_t size = cluster.size();

      stream << label << " " << size << std::endl;
      for (const LabeledDescriptor& labeled_descriptor : cluster)
      {
        const Descriptor& descriptor = labeled_descriptor.second;
        const int index = labeled_descriptor.first;
        stream << index << " ";

        for (const auto& value : descriptor)
        {
          stream << value << " ";
        }
        stream << "\n";
      }
    }
    std::cerr << "ClusterSpace serialization finished!" << std::endl;
    return stream;
  }
};


#endif // CLUSTERSPACE_H
