template <typename T>
Eigen::DenseIndex insertion_index(Eigen::Matrix<T, Eigen::Dynamic, 1> array,
                                  T value, Eigen::DenseIndex istart = 0) {
    assert(istart >= 0 && array.size() > istart);
    if (value <= array(0)) {
        return 0;
    } else if (value > array(array.size() - 1)) {
        return array.size();
    }

    Eigen::DenseIndex min_idx = (value > array(istart)) ? istart : 0;
    Eigen::DenseIndex max_idx = (value > array(istart)) ? array.size() : istart;
    Eigen::DenseIndex ins_idx = -1;

    // Narrow down the indices by golden ratio search
    float golden_ratio = (3.0f - sqrt(5.0f)) / 2.0f;
    while (max_idx - min_idx > 1) {
        ins_idx = min_idx + static_cast<Eigen::DenseIndex>(
                                ceil((max_idx - min_idx) * golden_ratio));
        if (value < array(ins_idx)) {
            max_idx = ins_idx;
        } else if (value > array(ins_idx)) {
            min_idx = ins_idx;
        } else {
            break;
        }
    }
    // Take the maximum index if we the loop terminated due to its condition
    // evaluating to false. Otherwise we don't change the value since we found
    // an exact hit.
    if (array(ins_idx) < value) {
        ins_idx = max_idx;
    }
    std::cout << "ins after loop: " << ins_idx << std::endl;

    // If there are several equal entries, take the leftmost one
    while (ins_idx != 0 && value == array(ins_idx - 1))
        --ins_idx;
    return ins_idx;
}
