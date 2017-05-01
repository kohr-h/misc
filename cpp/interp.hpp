#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


namespace interp {


template <typename T>
std::vector<Eigen::Index> sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                               Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2);

template <typename T>
std::vector<Eigen::Index> sorting_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> &v);

template <typename T>
std::vector<Eigen::Index> sorting_indices(std::vector<T> &v);

template <typename T>
std::vector<Eigen::Index> searchsorted(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                       Eigen::Matrix<T, Eigen::Dynamic, 1> v2);

} // namespace interp
