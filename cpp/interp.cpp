#include <iostream>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>

#include "interp.hpp"


template <typename T>
std::vector<Eigen::Index> interp::sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                               Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2) {
    std::vector<Eigen::Index> idcs;
    Eigen::Index i1 = 0, i2 = 0, ins_idx = 0;
    while (true) {
        if (i2 == v2.size()) break;

        if (i1 == v1.size()) {
            idcs.push_back(ins_idx);
            if (ins_idx < v1.size()) ++ins_idx;
            ++i2;
        } else {
            if (v1(i1) < v2(i2)) {
                ++i1;
                ++ins_idx;
            } else {
                idcs.push_back(ins_idx);
                ++i2;
            }
        }
    }
    return idcs;
}


template <typename T>
std::vector<Eigen::Index> interp::sorting_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> &v) {
    // Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes#12399290

    // Initialize original index locations 0, 1, ..., size-1
    std::vector<Eigen::Index> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](Eigen::Index i1, Eigen::Index i2) {return v(i1) < v(i2);});
    return idx;
}


template <typename T>
std::vector<Eigen::Index> interp::sorting_indices(std::vector<T> &v) {
    // Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes#12399290

    // Initialize original index locations 0, 1, ..., size-1
    std::vector<Eigen::Index> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](Eigen::Index i1, Eigen::Index i2) {return v[i1] < v[i2];});
    return idx;
}


template <typename T>
std::vector<Eigen::Index> interp::searchsorted(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                               Eigen::Matrix<T, Eigen::Dynamic, 1> v2) {
    // Same functionality as numpy.searchsorted
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
    // First sort the second array (passed by value), then return the indices of insertion of v2 into v1 such that
    // the resulting array remains sorted. For equal values, the smaller index is taken.

    // TODO: this sorts 3 times, do it in one go!
    // Get indices that sort v2 ("argsort"), and invert the mapping by sorting the index array
    std::vector<Eigen::Index> sort_idcs = sorting_indices(v2);
    std::vector<Eigen::Index> inv_sort_idcs = sorting_indices(sort_idcs);
    std::sort(v2.data(), v2.data() + v2.size());
    std::vector<Eigen::Index> merge_idcs = sorted_merge_indices(v1, v2);
    std::vector<Eigen::Index> ins_idcs(v2.size());
    assert(ins_idcs.size() == inv_sort_idcs.size() && ins_idcs.size() == merge_idcs.size());

    std::vector<Eigen::Index>::iterator it_inv_sort = inv_sort_idcs.begin(), it_ins = ins_idcs.begin();
    for (; it_inv_sort != inv_sort_idcs.end(); ++it_inv_sort, ++it_ins) {
        *it_ins = merge_idcs[*it_inv_sort];
    }
    return ins_idcs;
}
