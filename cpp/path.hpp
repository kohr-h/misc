#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

// -------------------------------------------------------------------------//

// Boundary conditions

enum class boundary_condition : int {
    zero,
    natural,
    clamp,
};

template <size_t ndim>
class BoundaryConditions {
  public:
    BoundaryConditions(boundary_condition bc = boundary_condition::natural)
        : BoundaryConditions<ndim>(bc, bc) {}
    BoundaryConditions(boundary_condition bc_left, boundary_condition bc_right);
    BoundaryConditions(
        std::vector<std::pair<boundary_condition, boundary_condition>> const&
            bcs)
        : bdry_conds_(bcs) {}

    std::vector<std::pair<boundary_condition, boundary_condition>>
    bdry_conds() const {
        return bdry_conds_;
    }

    std::pair<boundary_condition, boundary_condition>
    operator[](size_t index) const;

  private:
    std::vector<std::pair<boundary_condition, boundary_condition>> bdry_conds_;
};

template <size_t ndim>
std::ostream& operator<<(std::ostream& out,
                         BoundaryConditions<ndim> const& bcs);

// -------------------------------------------------------------------------//

// Path

template <size_t ndim>
class Path {
  public:
    Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
         boundary_condition bc = boundary_condition::natural);
    Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
         BoundaryConditions<ndim> const& bcs);
    Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
         Eigen::Matrix<float, 1, ndim> tang_left,
         Eigen::Matrix<float, 1, ndim> tang_right,
         boundary_condition bdry_cond = boundary_condition::clamp);
    Path(Eigen::Matrix<float, Eigen::Dynamic, ndim> const& nodes,
         Eigen::Matrix<float, 1, ndim> tang_left,
         Eigen::Matrix<float, 1, ndim> tang_right,
         BoundaryConditions<ndim> const& bcs);

    ~Path() {}

    const Eigen::Matrix<float, Eigen::Dynamic, ndim> nodes() const {
        return nodes_;
    }
    Eigen::DenseIndex num_nodes() const { return nodes().rows(); }
    Eigen::DenseIndex num_pieces() const { return num_nodes() - 1; }
    const Eigen::Matrix<float, Eigen::Dynamic, ndim> tangents() const {
        return tangents_;
    }
    const BoundaryConditions<ndim> bdry_conds() const { return bdry_conds_; }

    Eigen::Matrix<float, 1, ndim> operator()(float param) const;
    Eigen::Matrix<float, Eigen::Dynamic, ndim>
    operator()(Eigen::VectorXf const& params) const;

    Eigen::MatrixXf system_matrix(boundary_condition bc_left,
                                  boundary_condition bc_right) const;
    Eigen::VectorXf system_rhs(boundary_condition bc_left,
                               boundary_condition bc_right, size_t dim) const;
    Eigen::VectorXf arc_length_lin_approx(Eigen::VectorXf const& params) const;
    Eigen::VectorXf arc_length_params_lin_approx(size_t num_params) const;
    float total_length(size_t num_params) const;

  private:
    const Eigen::Matrix<float, Eigen::Dynamic, ndim> nodes_;
    const BoundaryConditions<ndim> bdry_conds_;
    Eigen::Matrix<float, Eigen::Dynamic, ndim> tangents_;
    std::vector<Eigen::PartialPivLU<Eigen::MatrixXf>> sys_matrix_decomps_;
    std::vector<Eigen::VectorXf> sys_rhs_;

    void compute_matrices_();
    void init_tangents_();
    void init_tangents_(Eigen::Matrix<float, 1, ndim> const& tang_left,
                        Eigen::Matrix<float, 1, ndim> const& tang_right);
    void init_rhs_();
    void compute_tangents_();
}; // class Path

template <size_t ndim>
std::ostream& operator<<(std::ostream& out, Path<ndim> const& p);
