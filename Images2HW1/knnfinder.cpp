#include "knnfinder.h"

#include <chrono>
#include <memory>
#include <cassert>
#include <fstream>
#include <random>

#include <QtGui/QApplication>

KnnFinder::KnnFinder(cv::Mat&& data) : m_data(std::move(data))
{
    if (!m_data.isContinuous())
    {
        std::cout << "Initial data not continious!" << std::endl;
        m_data = m_data.clone();
    }
}

double accuracy(const std::vector<int>& rightIndices, const std::vector<int>& guessedIndices)
{
    assert(rightIndices.size() == guessedIndices.size());

    double right_amount = rightIndices.size();
    double guessed = 0;

    for (size_t i = 0, size = rightIndices.size(); i < size; ++i)
    {
        if (rightIndices[i] == guessedIndices[i])
            ++guessed;
    }

    return guessed / right_amount;
}

std::function<double()> make_double_rand()
{
  std::random_device random_device;
  std::mt19937 mt(random_device());
  std::uniform_real_distribution<double> distribution(0, 1);
  std::function<double()> random_double = std::bind(distribution, mt);
  return random_double;
}

void KnnFinder::do_different_dimensions_search()
{
  using namespace cvflann;
  using namespace std::chrono;

  const int KNN = 10;
  const int N = 100;

  std::ofstream file("log.txt");
  cvflann::SearchParams search_params;

  std::function<double()> random_double = make_double_rand();

  std::vector<std::vector<int>> right_answers(N);
  std::vector<std::vector<double>> right_distances(N);

  for (int dimension = 500; dimension <= 500; dimension += 50)
  {
    std::vector<std::vector<double>> queries(N);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < dimension; ++j)
        queries[i].push_back(random_double());

    for (size_t index_number = 0; index_number < params.size(); ++index_number)
    {
      file << param_names[index_number] << " " << dimension << std::endl;

      cv::Mat continious_data = m_data;

      cv::flann::GenericIndex<cv::flann::L2<double>> index(continious_data, *params[index_number]);
      std::cout << param_names[index_number] << "with D = " << dimension << " built successfully!" << std::endl;

      double time_duration = 0;
      double current_accuracy = 0;

      for (int times = 0; times < N; ++times)
      {
        std::vector<double> query = queries[times];
        std::vector<int> indices(KNN);
        std::vector<double> distances(KNN);

        steady_clock::time_point now = steady_clock::now();
        index.knnSearch(query, indices, distances, KNN, search_params);
        steady_clock::time_point after = steady_clock::now();

        if (index_number == 0)
        {
          right_answers[times] = indices;
          right_distances[times] = distances;
        }
        else
        {
          current_accuracy += accuracy(right_answers[times], indices);
        }

        duration<double> time_span = duration_cast<duration<double>>(after - now);
        time_duration += time_span.count();
      }

      time_duration /= N;
      current_accuracy /= N;

      file << "Dim: " << dimension << " time: " << time_duration << " accuracy: " << current_accuracy << std::endl;
    }

    file << "------------------------------------------" << std::endl;
  }
}

void KnnFinder::do_different_size_search()
{
  using namespace std::chrono;
  std::ofstream log("log_different_sizes.txt");

  cvflann::SearchParams search_params;

  const int N = 100;
  const int KNN = 10;

  std::function<double()> random_double = make_double_rand();

  std::vector<std::vector<double>> queries(N);
  std::vector<std::vector<int>> right_indices(N);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < 500; ++j)
      queries[i].push_back(random_double());

  if (log.is_open())
  {
    for (size_t size = DATA_SIZE / 10; size <= DATA_SIZE; size += DATA_SIZE / 10)
    {
      cv::Mat data(m_data, cv::Range(0, size), cv::Range(0, m_data.cols));

      for (size_t index = 0; index < params.size(); ++index)
      {
        log << param_names[index] << " with " << size << " size:" << std::endl;

        cv::flann::GenericIndex<cvflann::L2<double>> generic_index(data, *params[index]);

        double duration = 0;
        double current_accuracy = 0;
        for (int times = 0; times < N; ++times)
        {
          std::vector<double> query = queries[times];
          std::vector<int> indices(KNN);
          std::vector<double> distances(KNN);

          steady_clock::time_point now = steady_clock::now();
          generic_index.knnSearch(query, indices, distances, KNN, search_params);
          steady_clock::time_point after = steady_clock::now();

          if (index == 0)
            right_indices[times] = indices;
          else
            current_accuracy += accuracy(right_indices[times], indices);

          std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(after - now);
          duration += time_span.count();
        }

        duration /= N;
        current_accuracy /= N;
        log << "search took " << duration << " seconds with accuracy: " << current_accuracy << std::endl;
      }
    }
  }
}
