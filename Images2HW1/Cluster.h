#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <array>
#include <utility>

template<class T, size_t Dim>
class Cluster
{
public:
  typedef std::array<T, Dim> Descriptor;
  typedef std::pair<int, Descriptor> LabeledDescriptor;
  typedef std::vector<LabeledDescriptor> DataStorage;

  Cluster(int label, const std::vector<T>& center) : m_label(label), m_center(center)
  {
  }

  Cluster(int label, DataStorage&& data)
    : m_label(label), m_data(std::move(data))
  {
  }

  void add_data(int index, const Descriptor& data)
  {
    m_data.push_back(std::make_pair(index, data));
  }

  void add_data(int index, Descriptor&& data)
  {
    m_data.push_back(std::make_pair(index, std::move(data)));
  }

  const DataStorage data() const
  {
    return m_data;
  }

  int label() const
  {
    return m_label;
  }

  typename DataStorage::const_iterator begin() const
  {
    return m_data.begin();
  }

  typename DataStorage::const_iterator end() const
  {
    return m_data.end();
  }

  size_t size() const
  {
    return m_data.size();
  }

  std::vector<size_t> indexes() const
  {
    std::vector<size_t> result;
    std::transform(m_data.begin(), m_data.end(), std::back_inserter(result), [&] (const LabeledDescriptor& desc) {
      return desc.first;
    });

    return result;
  }

private:
  int m_label;
  DataStorage m_data;
  std::vector<T> m_center;
};

#endif // CLUSTER_H
