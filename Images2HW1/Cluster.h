#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <array>
#include <utility>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/array.hpp>

template<class T, size_t Dim>
class Cluster
{
public:
  typedef std::array<T, Dim> Descriptor;
  typedef std::pair<int, Descriptor> LabeledDescriptor;
  typedef std::vector<LabeledDescriptor> DataStorage;

  Cluster() : m_label(0)
  {

  }

  Cluster(const Cluster& cluster) : m_label(cluster.label()),
                                    m_center(cluster.center()),
                                    m_data(cluster.data())
  {
  }

  Cluster(int label, const std::vector<T>& center) : m_label(label), m_center(center)
  {
  }

  Cluster(int label, DataStorage&& data)
    : m_label(label), m_data(std::move(data))
  {
  }

  Cluster(Cluster&& cluster) : m_label(cluster.label()), m_data(std::move(cluster.data()))
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

  DataStorage data() const
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

  const std::vector<T>& center() const
  {
      return m_center;
  }

  std::vector<size_t> indexes() const
  {
    std::vector<size_t> result;
    std::transform(m_data.begin(), m_data.end(), std::back_inserter(result), [&] (const LabeledDescriptor& desc) {
      return desc.first;
    });

    return result;
  }

  template <class Archive>
  void serialize(Archive& archive, const int)
  {
    archive & m_label;
    archive & m_center;
    archive & m_data;
  }

private:
  int m_label;
  std::vector<T> m_center;
  DataStorage m_data;
};

namespace boost
{
namespace serialization
{
  template <class Archive, class T, size_t l>
  void serialize(Archive& ar, std::array<T, l>& array, const unsigned int)
  {
    ar & boost::serialization::make_array(array.data(), array.size());
  }
}
}

#endif // CLUSTER_H
