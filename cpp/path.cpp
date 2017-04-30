#include <iostream>
#include <stdexcept>
#include <eigen3/Eigen/Dense>

# include "path.hpp"


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
    size_t n_nodes = num_nodes();
    Eigen::MatrixXf sys_mat = Eigen::MatrixXf::Zero(n_nodes, n_nodes);

    for (size_t row = 1; row < n_nodes - 1; ++row) {
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
    size_t n_nodes = nds.rows();
    Eigen::VectorXf sys_rhs(n_nodes);

    for (size_t i = 1; i < n_nodes - 1; ++i) {
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
    Eigen::VectorXf alen_parms(params.size());
    Eigen::VectorXf alens = arc_length_lin_approx(params);
    float total_len = alens(alens.size() - 1);

    for (int i = 1; i < params.size(); ++i) {
        // Interpolate the target arc length in the computed arc lengths.
        unsigned int interp_idx = static_cast<unsigned int>(floor());
        float s = param - piece;  // Normalized to [0, 1]

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
    Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(50, 0, 4);
    std::cout << p(params) << std::endl << "-----" << std::endl;
    std::cout << p.arc_length_lin_approx(params) << std::endl << "-----" << std::endl;
}
