#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


template <typename T>
std::vector<size_t> sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                         Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2);
template <typename T>
std::vector<size_t> sorting_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> &v);
template <typename T>
std::vector<size_t> sorting_indices(std::vector<T> &v);
template <typename T>
std::vector<size_t> searchsorted(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                 Eigen::Matrix<T, Eigen::Dynamic, 1> v2);
