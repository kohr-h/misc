#include <eigen3/Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>

#include "glm/glm.hpp"
#include "path.hpp"

// -------------------------------------------------------------------------//

// Helper functions

template <typename T>
std::vector<Eigen::DenseIndex>
sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const& v1,
                     Eigen::Matrix<T, Eigen::Dynamic, 1> const& v2) {
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

// -------------------------------------------------------------------------//

// Boundary condition handling

template <size_t ndim>
BoundaryConditions<ndim>::BoundaryConditions(boundary_condition bc_left,
                                             boundary_condition bc_right) {
    bdry_conds_.resize(0);
    for (size_t dim = 0; dim < ndim; dim++) {
        bdry_conds_.push_back(std::pair<boundary_condition, boundary_condition>(
            bc_left, bc_right));
    }
}

std::ostream& operator<<(std::ostream& out, boundary_condition bc) {
    switch (bc) {
    case boundary_condition::natural:
        out << "n";
        break;
    case boundary_condition::clamp:
        out << "c";
        break;
    case boundary_condition::zero:
        out << "0";
        break;
    }
    return out;
}

template <size_t ndim>
std::ostream& operator<<(std::ostream& out,
                         BoundaryConditions<ndim> const& bcs) {
    std::cout << "Boundary conditions" << std::endl;
    for (auto bc : bcs.bdry_conds()) {
        std::cout << bc.first << " " << bc.second << std::endl;
    }
    return out;
}

template <size_t ndim>
std::pair<boundary_condition, boundary_condition> BoundaryConditions<ndim>::
operator[](size_t index) const {
    return bdry_conds_[index];
}

// -------------------------------------------------------------------------//

// Path constructors

template <size_t ndim>
Path<ndim>::Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
                 boundary_condition bc)
    : nodes_(nodes), bdry_conds_(bc) {
    assert(nodes.rows() >= 2);
    compute_matrices_();
    init_tangents_();
    init_rhs_();
    compute_tangents_();
}

template <size_t ndim>
Path<ndim>::Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
                 BoundaryConditions<ndim> const& bcs)
    : nodes_(nodes), bdry_conds_(bcs) {
    assert(nodes.rows() >= 2);
    compute_matrices_();
    init_tangents_();
    init_rhs_();
    compute_tangents_();
}

template <size_t ndim>
Path<ndim>::Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
                 Eigen::Matrix<float, 1, ndim> tang_left,
                 Eigen::Matrix<float, 1, ndim> tang_right,
                 boundary_condition bc)
    : nodes_(nodes), bdry_conds_(bc) {
    assert(nodes.rows() >= 2);
    compute_matrices_();
    init_tangents_(tang_left, tang_right);
    init_rhs_();
    compute_tangents_();
}

template <size_t ndim>
Path<ndim>::Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
                 Eigen::Matrix<float, 1, ndim> tang_left,
                 Eigen::Matrix<float, 1, ndim> tang_right,
                 BoundaryConditions<ndim> const& bcs)
    : nodes_(nodes), bdry_conds_(bcs) {
    assert(nodes.rows() >= 2);
    compute_matrices_();
    init_tangents_(tang_left, tang_right);
    init_rhs_();
    compute_tangents_();
}

// Private functions (for initialization)

template <size_t ndim>
void Path<ndim>::compute_matrices_() {
    for (size_t dim = 0; dim < ndim; ++dim) {
        Eigen::MatrixXf sys_mat =
            system_matrix(bdry_conds_[dim].first, bdry_conds_[dim].second);
        Eigen::PartialPivLU<Eigen::MatrixXf> decomp = sys_mat.partialPivLu();
        sys_matrix_decomps_.push_back(decomp);
    }
}

template <size_t ndim>
void Path<ndim>::init_tangents_() {
    const Eigen::Matrix<float, 1, ndim> tang_left =
        nodes_.row(1) - nodes_.row(0);
    const Eigen::Matrix<float, 1, ndim> tang_right =
        nodes_.row(num_nodes() - 1) - nodes_.row(num_nodes() - 2);
    init_tangents_(tang_left, tang_right);
}

template <size_t ndim>
void Path<ndim>::init_tangents_(
    Eigen::Matrix<float, 1, ndim> const& tang_left,
    Eigen::Matrix<float, 1, ndim> const& tang_right) {
    tangents_.resize(num_nodes(), ndim);
    tangents_.row(0) = tang_left;
    tangents_.row(num_nodes() - 1) = tang_right;
}

template <size_t ndim>
void Path<ndim>::init_rhs_() {
    for (size_t dim = 0; dim < ndim; ++dim) {
        sys_rhs_.push_back(this->system_rhs(bdry_conds_[dim].first,
                                            bdry_conds_[dim].second, dim));
    }
}

template <size_t ndim>
void Path<ndim>::compute_tangents_() {
    for (size_t dim = 0; dim < ndim; ++dim) {
        tangents_.col(dim) =
            this->sys_matrix_decomps_[dim].solve(this->sys_rhs_[dim]);
    }
}

// Other member functions

template <size_t ndim>
Eigen::MatrixXf Path<ndim>::system_matrix(boundary_condition bc_left,
                                          boundary_condition bc_right) const {
    // Return the matrix of the equation system for the tangents of the path at
    // the given nodes.
    Eigen::DenseIndex n_nodes = num_nodes();
    Eigen::MatrixXf sys_mat = Eigen::MatrixXf::Zero(n_nodes, n_nodes);

    // All rows except the first and last have an entry 4.0 on the diagonal and
    // 1.0 on the first off-diagonals,
    // elsewhere 0.0.
    for (Eigen::DenseIndex row = 1; row < n_nodes - 1; ++row) {
        sys_mat(row, row) = 4.0;
        sys_mat(row, row - 1) = 1.0;
        sys_mat(row, row + 1) = 1.0;
    }

    switch (bc_left) {
    case boundary_condition::zero:
    case boundary_condition::clamp:
        // Just an identity equation, keeps original left tangent
        sys_mat(0, 0) = 1.0;
        break;
    case boundary_condition::natural:
        sys_mat(0, 0) = 2.0;
        sys_mat(0, 1) = 1.0;
        break;
    default:
        std::cout << "Invalid left boundary condition " << bc_left << "!";
    }
    switch (bc_right) {
    case boundary_condition::zero:
    case boundary_condition::clamp:
        // Just an identity equation, keeps original right tangent
        sys_mat(n_nodes - 1, n_nodes - 1) = 1.0;
        break;
    case boundary_condition::natural:
        sys_mat(n_nodes - 1, n_nodes - 1) = 2.0;
        sys_mat(n_nodes - 1, n_nodes - 2) = 1.0;
        break;
    default:
        std::cout << "Invalid right boundary condition " << bc_right << "!";
    }
    return sys_mat;
}

template <size_t ndim>
Eigen::VectorXf Path<ndim>::system_rhs(boundary_condition bc_left,
                                       boundary_condition bc_right,
                                       size_t dim) const {
    // Return the right-hand side of the equation system for the tangents of the
    // path at the given nodes.
    Eigen::Matrix<float, Eigen::Dynamic, ndim> nds = nodes();
    Eigen::DenseIndex n_nodes = nds.rows();
    Eigen::VectorXf sys_rhs(n_nodes);

    // All equations except the first and last have 3.0 times the difference
    // between the next and previous nodes
    // as right-hand sides.
    for (Eigen::DenseIndex i = 1; i < n_nodes - 1; ++i) {
        sys_rhs(i) = 3.0 * (nds(i + 1, dim) - nds(i - 1, dim));
    }

    switch (bc_left) {
    case boundary_condition::zero:
        sys_rhs(0) = 0.0;
        break;
    case boundary_condition::clamp:
        sys_rhs(0) = tangents()(0, dim);
        break;
    case boundary_condition::natural:
        sys_rhs(0) = 3.0 * (nds(1, dim) - nds(0, dim));
        break;
    default:
        std::cout << "Invalid left boundary condition " << bc_left << "!";
    }
    switch (bc_right) {
    case boundary_condition::zero:
        sys_rhs(n_nodes - 1) = 0.0;
        break;
    case boundary_condition::clamp:
        sys_rhs(n_nodes - 1) = nds(n_nodes - 1, dim);
        break;
    case boundary_condition::natural:
        sys_rhs(n_nodes - 1) =
            3.0 * (nds(n_nodes - 1, dim) - nds(n_nodes - 2, dim));
        break;
    default:
        std::cout << "Invalid right boundary condition " << bc_right << "!";
    }
    return sys_rhs;
}

template <size_t ndim>
Eigen::VectorXf
Path<ndim>::arc_length_lin_approx(Eigen::VectorXf const& params) const {
    // Compute an approximation to the arc lengths at `params` using a piecewise
    // constant path instead of the
    // actual spline.
    // This is done by computing the path points at `params` and then
    // accumulating the lengths of the differences
    // of the points.
    assert((params.array() >= 0).all() &&
           (params.array() <= num_pieces()).all());
    Eigen::Matrix<float, Eigen::Dynamic, ndim> path_pts =
        this->operator()(params);
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
Eigen::VectorXf
Path<ndim>::arc_length_params_lin_approx(size_t num_params) const {
    // Interpolate the parameters that yield points that are equally spaced with
    // respect to arc length.
    // This is the discrete counterpart of reparametrization with respect to arc
    // length.
    //
    // See
    // https://en.wikipedia.org/wiki/Differential_geometry_of_curves#Length_and_natural_parametrization
    // for details.
    Eigen::VectorXf alen_params(num_params);
    Eigen::VectorXf params =
        Eigen::VectorXf::LinSpaced(num_params, 0.0f, num_pieces());
    Eigen::VectorXf alens = arc_length_lin_approx(params);
    Eigen::VectorXf target_alens =
        Eigen::VectorXf::LinSpaced(num_params, 0.0f, total_length(num_params));
    std::vector<Eigen::DenseIndex> pieces =
        sorted_merge_indices(alens, target_alens);
    for (size_t i = 0; i < num_params; ++i) {
        // Interpolate the target arc length in the computed arc lengths.
        float s; // Normalized distance to target node to the left
        if (pieces[i] == 0) {
            s = 0.0f;
            pieces[i] = 1;
        } else if (static_cast<size_t>(pieces[i]) == num_params) {
            s = 1.0f;
            pieces[i] = num_params - 1;
        } else {
            s = (target_alens(i) - alens(pieces[i] - 1)) /
                (alens(pieces[i]) - alens(pieces[i] - 1));
        }
        alen_params(i) =
            (1.0f - s) * params(pieces[i] - 1) + s * params(pieces[i]);
    }
    return alen_params;
}

template <size_t ndim>
float Path<ndim>::total_length(size_t num_params) const {
    Eigen::VectorXf params =
        Eigen::VectorXf::LinSpaced(num_params, 0, num_pieces());
    Eigen::VectorXf alens = arc_length_lin_approx(params);
    return alens(alens.size() - 1);
}

// Operator overloads for Path

template <size_t ndim>
std::ostream& operator<<(std::ostream& out, Path<ndim> const& p) {
    out << "Path with nodes" << std::endl
        << p.nodes() << std::endl
        << "and boundary conditions" << std::endl
        << p.bdry_conds() << std::endl;
    return out;
}

template <size_t ndim>
Eigen::Matrix<float, 1, ndim> Path<ndim>::operator()(float param) const {
    assert((param >= 0) && (param <= num_pieces()));

    auto piece = static_cast<Eigen::DenseIndex>(floor(param));
    float s = param - piece; // Normalized to [0, 1]
    if (piece == num_pieces()) {
        return nodes().row(num_pieces());
    } // right end of the parameter range

    // Compute path points as
    //     path(s) = p[k] + s(p[k+1] - p[k]) + s(1-s)((1-s)a + sb),
    // where p = nodes, s = param in [0, 1], and a, b as below.
    // See https://en.wikipedia.org/wiki/Spline_interpolation
    Eigen::Matrix<float, 1, ndim> diff =
        nodes().row(piece + 1) - nodes().row(piece);
    Eigen::Matrix<float, 1, ndim> a = tangents().row(piece) - diff;
    Eigen::Matrix<float, 1, ndim> b = diff - tangents().row(piece + 1);
    return nodes().row(piece) + s * diff +
           s * (1.0 - s) * ((1.0 - s) * a + s * b);
}

template <size_t ndim>
Eigen::Matrix<float, Eigen::Dynamic, ndim> Path<ndim>::
operator()(Eigen::VectorXf const& params) const {
    Eigen::Matrix<float, Eigen::Dynamic, ndim> points(params.size(), ndim);
    for (int i = 0; i < params.size(); ++i) {
        points.row(i) = this->operator()(params(i));
    }
    return points;
}

// -------------------------------------------------------------------------//

// Playground

int main() {
    Eigen::MatrixXf v(4, 2);
    v << 0, 0, 1, 1, -1, 2, 0, 2;
    v.resize(5, 2);
    v << 0, 0, 1, 1, -1, 2, 0, 2, 5, 5;
    Path<2> p(v);
    // std::cout << p << std::endl << "-----" << std::endl;
    Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(10, 0, 4);
    std::cout << "total length: " << p.total_length(50) << std::endl
              << "-----" << std::endl;
    auto alen_params = p.arc_length_params_lin_approx(20);
    std::cout << "arc length params:" << std::endl
              << alen_params.transpose() << std::endl
              << "-----" << std::endl;
    std::cout << "arc length at new parameters:" << std::endl
              << p.arc_length_lin_approx(alen_params).transpose() << std::endl;

    glm::vec4 vec(1.0f, 0.0f, 1.2f, 4.1f);
    std::cout << "vec:" << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    BoundaryConditions<2> bcs(boundary_condition::natural);
    std::cout << bcs << std::endl;
}
