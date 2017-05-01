#include <iostream>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>

#include "path.hpp"


// TODO: move to separate file
// ---------------------------------------------------------------------------


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

// ---------------------------------------------------------------------------


template <size_t ndim>
std::ostream &operator<<(std::ostream &out, Path<ndim> &p) {
    out << "Path with nodes" << std::endl
        << p.nodes() << std::endl
        << "and boundary conditions" << std::endl
        << p.bdry_conds() << std::endl;
    return out;
}


template <size_t ndim>
const Eigen::MatrixXf Path<ndim>::system_matrix(bc_t bc_left, bc_t bc_right) {
    Eigen::Index n_nodes = num_nodes();
    Eigen::MatrixXf sys_mat = Eigen::MatrixXf::Zero(n_nodes, n_nodes);

    for (Eigen::Index row = 1; row < n_nodes - 1; ++row) {
        sys_mat(row, row) = 4.0;
        sys_mat(row, row - 1) = 1.0;
        sys_mat(row, row + 1) = 1.0;
    }

    switch(bc_left) {
        case BC_ZERO:
        case BC_CLAMP:
            sys_mat(0, 0) = 1.0; break;
        case BC_NATURAL:
            sys_mat(0, 0) = 2.0;
            sys_mat(0, 1) = 1.0;
            break;
        default:
            std::cout << "Invalid left boundary condition " << bc_left << "!";
    }
    switch(bc_right) {
        case BC_ZERO:
        case BC_CLAMP:
            sys_mat(n_nodes - 1, n_nodes - 1) = 1.0;
            break;
        case BC_NATURAL:
            sys_mat(n_nodes - 1, n_nodes - 1) = 2.0;
            sys_mat(n_nodes - 1, n_nodes - 2) = 1.0;
            break;
        default:
            std::cout << "Invalid right boundary condition " << bc_right << "!";
    }
    return sys_mat;
}


template <size_t ndim>
const Eigen::VectorXf Path<ndim>::system_rhs(bc_t bc_left, bc_t bc_right, size_t dim) {
    Vectors<ndim> nds = nodes();
    Eigen::Index n_nodes = nds.rows();
    Eigen::VectorXf sys_rhs(n_nodes);

    for (Eigen::Index i = 1; i < n_nodes - 1; ++i) {
        sys_rhs(i) = 3.0 * (nds(i + 1, dim) - nds(i - 1, dim));
    }

    switch(bc_left) {
        case BC_ZERO:
            sys_rhs(0) = 0.0; break;
        case BC_CLAMP:
            sys_rhs(0) = tangents()(0, dim); break;
        case BC_NATURAL:
            sys_rhs(0) = 3.0 * (nds(1, dim) - nds(0, dim)); break;
        default:
            std::cout << "Invalid left boundary condition " << bc_left << "!";
    }
    switch(bc_right) {
        case BC_ZERO:
            sys_rhs(n_nodes - 1) = 0.0; break;
        case BC_CLAMP:
            sys_rhs(n_nodes - 1) = nds(n_nodes - 1, dim); break;
        case BC_NATURAL:
            sys_rhs(n_nodes - 1) = 3.0 * (nds(n_nodes - 1, dim) - nds(n_nodes - 2, dim)); break;
        default:
            std::cout << "Invalid right boundary condition " << bc_right << "!";
    }
    return sys_rhs;
}


template <size_t ndim>
const Eigen::VectorXf Path<ndim>::arc_length_lin_approx(Eigen::VectorXf params) {
    assert((params.array() >= 0).all() && (params.array() <= num_pieces()).all());

    // Compute the path points parametrized by `params`. Then compute the arc lengths by accumulating
    // the lengths of the differences of the points.
    Vectors<ndim> path_pts = this->operator()(params);
    Eigen::VectorXf alens(params.size());
    alens(0) = 0.0;
    for (int i = 1; i < params.size(); ++i) {
        alens(i) = (path_pts.row(i) - path_pts.row(i - 1)).norm();
    }
    float acc = 0.0;
    for (int i = 0; i < params.size(); ++i) {
        acc += alens(i);
        alens(i) = acc;
    }
    // Update total length if approximation was finer than the initial one and the parameters went all the way to
    // the end
    if (static_cast<size_t>(params.size()) > NUM_PTS_PER_PIECE * num_pieces() &&
            fabsf(params(params.size() - 1) - num_pieces()) < 1e-6) {
        total_length_ = alens(alens.size() - 1);
    }
    return alens;
}


template <size_t ndim>
const Eigen::VectorXf Path<ndim>::arc_length_params_lin_approx(Eigen::VectorXf params, Eigen::VectorXf arc_lengths) {
    assert((params.array() >= 0).all() && (params.array() <= num_pieces()).all());

    // Compute the arc lengths using path points at `params`. Then interpolate the parameters such that the
    // corresponding path points yield the given `arc_lengths`.
    // This is the discrete counterpart of reparametrization with respect to arc length.
    //
    // See https://en.wikipedia.org/wiki/Differential_geometry_of_curves#Length_and_natural_parametrization for details.
    Eigen::Index num_params = params.size();
    Eigen::VectorXf alen_params(num_params);
    Eigen::VectorXf alens = arc_length_lin_approx(params);
    std::cout << alens.transpose() << std::endl;
    assert((arc_lengths.array() <= total_length()).all());
    std::vector<Eigen::Index> pieces = interp::searchsorted(arc_lengths, alens);
    std::cout << "alive" << std::endl;
    for (auto v: pieces) { std::cout << v << " "; }
    std::cout << std::endl;

    for (Eigen::Index i = 0; i < num_params; ++i) {
        // Interpolate the target arc length in the computed arc lengths.
        float s = alens(i) - pieces[i];  // Normalized to [0, 1]
        alen_params(i) = (1.0f - s) * params(pieces[i]) + s * params(pieces[i + 1]);
    }
    return alen_params;
}


template <size_t ndim>
Vector<ndim> Path<ndim>::operator()(float param) {
    assert((param >= 0) && (param <= num_pieces()));

    unsigned int piece = static_cast<unsigned int>(floor(param));
    float s = param - piece;  // Normalized to [0, 1]
    if (piece == num_pieces()) { return nodes().row(num_pieces()); }  // right end of the parameter range

    // Compute spline points as
    //     spline(s) = p[k] + s(p[k+1] - p[k]) + s(1-s)((1-s)a + sb),
    // where p = nodes, s = param in [0, 1], and a, b as below.
    // See https://en.wikipedia.org/wiki/Spline_interpolation
    Vector<ndim> diff = nodes().row(piece + 1) - nodes().row(piece);
    Vector<ndim> a = tangents().row(piece) - diff;
    Vector<ndim> b = diff - tangents().row(piece + 1);
    return nodes().row(piece) + s * diff + s * (1.0 - s) * ((1.0 - s) * a + s * b);
}


template <size_t ndim>
Vectors<ndim> Path<ndim>::operator()(Eigen::VectorXf params) {
    Vectors<ndim> points(params.size());
    for (int i = 0; i < params.size(); ++i) {
        points.row(i) = this->operator()(params(i));
    }
    return points;
}


// TODO: move to separate file
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------

int main () {
    Vectors<2> v(4);
    v << 0, 0,
         1, 1,
        -1, 2,
         0, 2;
    v.resize(5);
    v << 0, 0,
         1, 1,
        -1, 2,
         0, 2,
         5, 5;
    std::cout << v << std::endl << "-----" << std::endl;
    Path<2> p(v);
    std::cout << p << std::endl << "-----" << std::endl;
    BdryConds<2> bcs(BC_ZERO);
    std::cout << bcs << std::endl << "-----" << std::endl;

    Eigen::MatrixXf m = p.system_matrix(BC_CLAMP, BC_NATURAL);
    std::cout << m << std::endl << "-----" << std::endl;

    std::cout << p.tangents() << std::endl << "-----" << std::endl;
    std::cout << p(2.5) << std::endl << "-----" << std::endl;
    Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(10, 0, 4);
    std::cout << p(params) << std::endl << "-----" << std::endl;
    std::cout << p.arc_length_lin_approx(params) << std::endl << "-----" << std::endl;
    std::cout << p.total_length() << std::endl << "-----" << std::endl;
    Eigen::VectorXf target_alens = Eigen::VectorXf::LinSpaced(10, 0, 11);
    std::cout << "==============" << std::endl;
    std::cout << p.arc_length_params_lin_approx(params, target_alens) << std::endl << "-----" << std::endl;
}
