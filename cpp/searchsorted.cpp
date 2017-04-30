#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


template <typename T>
void print(std::vector<T> const &v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size() - 1; ++i) {
        std::cout << v[i] << ", ";
    }
    std::cout << v[v.size() - 1] << "]" << std::endl;
}


template <typename T>
std::vector<size_t> merge_indices(std::vector<T> const &v1, std::vector<T> const &v2) {
    std::vector<size_t> idcs;

    typename std::vector<T>::const_iterator it1 = v1.begin();
    typename std::vector<T>::const_iterator it2 = v2.begin();

    for (size_t index = 0;;) {
        if (it2 == v2.end()) break;

        if (it1 == v1.end()) {
            idcs.push_back(index);
            ++it2;
        } else {
            if (*it1 < *it2) {
                ++it1;
                if (index <= v1.size()) {
                    ++index;
                }
            } else {
                idcs.push_back(index);
                ++it2;
            }
        }
    }
    return idcs;
}


template <typename T>
std::vector<size_t> sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                         Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2) {
    std::vector<size_t> idcs;

    int i1 = 0, i2 = 0, ins_idx = 0;

    while (true) {
        if (i2 == v2.size()) break;

        if (i1 == v1.size() - 1) {
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
std::vector<size_t> sorting_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> &v) {
    // Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes#12399290

    // Initialize original index locations 0, 1, ..., size-1
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});
    return idx;
}


template <typename T>
std::vector<size_t> sorting_indices(std::vector<T> &v) {
    // Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes#12399290

    // Initialize original index locations 0, 1, ..., size-1
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}


template <typename T>
std::vector<size_t> searchsorted(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                 Eigen::Matrix<T, Eigen::Dynamic, 1> v2) {
    // Same functionality as numpy.searchsorted
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
    // First sort the second array (passed by value), then return the indices of insertion of v2 into v1 such that
    // the resulting array remains sorted. For equal values, the smaller index is taken.

    // TODO: this sorts 3 times, do it in one go!
    // Get indices that sort v2 ("argsort"), and invert the mapping by sorting the index array
    std::vector<size_t> sort_idcs = sorting_indices(v2);
    std::vector<size_t> inv_sort_idcs = sorting_indices(sort_idcs);
    std::sort(v2.data(), v2.data() + v2.size());
    std::vector<size_t> merge_idcs = sorted_merge_indices(v1, v2);
    std::vector<size_t> ins_idcs(v2.size());
    print(sort_idcs);
    print(inv_sort_idcs);
    print(merge_idcs);
    assert(ins_idcs.size() == inv_sort_idcs.size() && ins_idcs.size() == merge_idcs.size());

    std::vector<size_t>::iterator it_inv_sort = inv_sort_idcs.begin(), it_ins = ins_idcs.begin();
    for (; it_inv_sort != inv_sort_idcs.end(); ++it_inv_sort, ++it_ins) {
        *it_ins = merge_idcs[*it_inv_sort];
    }
    return ins_idcs;
}


int main() {
    Eigen::VectorXi v1(4), v2(7);
    v1 << 0, 2, 3, 5;
    v2 << 0, 3, 7, 3, 2, 6, 1;

    std::vector<size_t> indices = searchsorted(v1, v2);
    print(indices);
}
