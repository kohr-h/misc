#include <iostream>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>

#include "path.hpp"


template <typename T>
std::vector<Eigen::DenseIndex> sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                                    Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2) {
    std::vector<Eigen::DenseIndex> idcs;
    Eigen::DenseIndex i1 = 0, i2 = 0, ins_idx = 0;
    while (true) {
        if (i2 == v2.size()) {
            break;
        }

        if (i1 == v1.size()) {
            idcs.push_back(ins_idx);
            if (ins_idx < v1.size()) {
                ++ins_idx;
            }
            ++i2;
        } else {
            if (v1(i1) < v2(i2)) {
                ++ins_idx;
                ++i1;
            } else {
                idcs.push_back(ins_idx);
                ++i2;
            }
        }
    }
    return idcs;
}


template <size_t ndim>
std::ostream &operator<<(std::ostream &out, Path<ndim> const &p) {
    out << "Path with nodes" << std::endl
        << p.nodes() << std::endl
        << "and boundary conditions" << std::endl
        << p.bdry_conds() << std::endl;
    return out;
}


template <size_t ndim>
Eigen::MatrixXf Path<ndim>::system_matrix(bc_t bc_left, bc_t bc_right) const {
    Eigen::DenseIndex n_nodes = num_nodes();
    Eigen::MatrixXf sys_mat = Eigen::MatrixXf::Zero(n_nodes, n_nodes);

    for (Eigen::DenseIndex row = 1; row < n_nodes - 1; ++row) {
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
Eigen::VectorXf Path<ndim>::system_rhs(bc_t bc_left, bc_t bc_right, size_t dim) const {
    Vectors<ndim> nds = nodes();
    Eigen::DenseIndex n_nodes = nds.rows();
    Eigen::VectorXf sys_rhs(n_nodes);

    for (Eigen::DenseIndex i = 1; i < n_nodes - 1; ++i) {
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
Eigen::VectorXf Path<ndim>::arc_length_lin_approx(Eigen::VectorXf const &params) const {
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
    return alens;
}


template <size_t ndim>
Eigen::VectorXf Path<ndim>::arc_length_params_lin_approx(size_t num_params) const {
    // Interpolate the parameters that yield points that are equally spaced with respect to arc length.
    // This is the discrete counterpart of reparametrization with respect to arc length.
    //
    // See https://en.wikipedia.org/wiki/Differential_geometry_of_curves#Length_and_natural_parametrization for details.
    Eigen::VectorXf alen_params(num_params);
    Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(num_params, 0.0f, num_pieces());
    Eigen::VectorXf alens = arc_length_lin_approx(params);
    Eigen::VectorXf target_alens = Eigen::VectorXf::LinSpaced(num_params, 0.0f, total_length(num_params));
    std::vector<Eigen::DenseIndex> pieces = sorted_merge_indices(alens, target_alens);

    std::cout << "linspaced params:" << std::endl << params.transpose() << std::endl;
    std::cout << "target_alens:" << std::endl << target_alens.transpose() << std::endl;
    std::cout << "alens:" << std::endl << alens.transpose() << std::endl;

    std::cout << "pieces:" << std::endl;
    for (auto v: pieces) { std::cout << v << " "; }
    std::cout << std::endl;

    std::cout << "s values:" << std::endl;
    for (size_t i = 0; i < num_params; ++i) {
        // Interpolate the target arc length in the computed arc lengths.
        float s;  // Normalized distance to target node to the left
        if (pieces[i] == 0) {
            s = 0.0f;
            pieces[i] = 1;
        } else if (static_cast<size_t>(pieces[i]) == num_params) {
            s = 1.0f;
            pieces[i] = num_params - 1;
        } else {
            s = (target_alens(i) - alens(pieces[i] - 1)) / (alens(pieces[i]) - alens(pieces[i] - 1));
        }
        std::cout << s << " ";
        alen_params(i) = (1.0f - s) * params(pieces[i] - 1) + s * params(pieces[i]);
    }
    std::cout << std::endl;
    return alen_params;
}


template <size_t ndim>
Vector<ndim> Path<ndim>::operator()(float param) const {
    assert((param >= 0) && (param <= num_pieces()));

    auto piece = static_cast<Eigen::DenseIndex>(floor(param));
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
Vectors<ndim> Path<ndim>::operator()(Eigen::VectorXf const &params) const {
    Vectors<ndim> points(params.size());
    for (int i = 0; i < params.size(); ++i) {
        points.row(i) = this->operator()(params(i));
    }
    return points;
}


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
    Path<2> p(v);
    // std::cout << p << std::endl << "-----" << std::endl;
    Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(10, 0, 4);
    std::cout << "total length: " << p.total_length(50) << std::endl << "-----" << std::endl;
    auto alen_params = p.arc_length_params_lin_approx(20);
    std::cout << "arc length params:" << std::endl << alen_params.transpose() << std::endl << "-----" << std::endl;
    std::cout << "arc length at new parameters:" << std::endl << p.arc_length_lin_approx(alen_params).transpose()
              << std::endl;
}
