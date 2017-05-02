#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>


// Flags for boundary conditions
// Since Eigen arrays don't work well with enum types, we just take integer flags
typedef int bc_t;
static const bc_t BC_T_START = 0, BC_ZERO = 0, BC_NATURAL = 1, BC_CLAMP = 2, BC_T_END = 3;

static const size_t NUM_PTS_PER_PIECE = 50;  // Discretization per path piece to compute total length

// Partially specialized "typedefs"
// The `<type>Impl` templates are to be subclassed by the intended final `<type>` can overload constructors and
// other stuff.
template <size_t ndim> using
BdryCondsImpl = Eigen::Array<bc_t, 2, ndim>;

template <size_t ndim> using
Vector = Eigen::Matrix<float, 1, ndim>;

template <size_t ndim> using
VectorsImpl = Eigen::Matrix<float, Eigen::Dynamic, ndim>;


template <size_t ndim>
class Vectors : public VectorsImpl<ndim> {
  public:
    Vectors() : VectorsImpl<ndim>() {}
    template<typename OtherDerived>
    Vectors(Eigen::MatrixBase<OtherDerived> const &other) : VectorsImpl<ndim>(other) {}
    Vectors(size_t rows) : VectorsImpl<ndim>(rows, ndim) {}

    void resize(size_t new_rows) { VectorsImpl<ndim>::resize(new_rows, ndim); }

    template<typename OtherDerived>
    Vectors& operator= (Eigen::MatrixBase <OtherDerived> const &other) {
        this->VectorsImpl<ndim>::operator=(other);
        return *this;
    }
};  // class Vectors


template <size_t ndim>
class BdryConds : public BdryCondsImpl<ndim> {
  public:
    BdryConds() : BdryConds<ndim>(BC_NATURAL) {}
    BdryConds(bc_t bdry_cond) {
        assert((bdry_cond >= BC_T_START) && (bdry_cond < BC_T_END));
        this->setConstant(bdry_cond);
    }
};  // class BdryConds


template <typename T>
std::vector<Eigen::DenseIndex> sorted_merge_indices(Eigen::Matrix<T, Eigen::Dynamic, 1> const &v1,
                                                    Eigen::Matrix<T, Eigen::Dynamic, 1> const &v2);


template <size_t ndim>
class Path {
  public:
    Path(Vectors<ndim> const &nodes, bc_t bdry_cond = BC_NATURAL) : nodes_(nodes), bdry_conds_(bdry_cond) {
        assert(nodes.rows() >= 2);
        compute_matrices_();
        init_tangents_();
        init_rhs_();
        compute_tangents_();
    }

    Path(Vectors<ndim> const &nodes, BdryConds<ndim> const &bdry_conds) : nodes_(nodes), bdry_conds_(bdry_conds) {
        assert(nodes.rows() >= 2);
        compute_matrices_();
        init_tangents_();
        init_rhs_();
        compute_tangents_();
    }

    Path(Vectors<ndim> const &nodes, Vector<ndim> tang_left, Vector<ndim> tang_right, bc_t bdry_cond = BC_CLAMP)
            : nodes_(nodes), bdry_conds_(bdry_cond) {
        assert(nodes.rows() >= 2);
        compute_matrices_();
        init_tangents_(tang_left, tang_right);
        init_rhs_();
        compute_tangents_();
    }

    Path(Vectors<ndim> const &nodes, Vector<ndim> tang_left, Vector<ndim> tang_right, BdryConds<ndim> const &bdry_conds)
            : nodes_(nodes), bdry_conds_(bdry_conds) {
        assert(nodes.rows() >= 2);
        compute_matrices_();
        init_tangents_(tang_left, tang_right);
        init_rhs_();
        compute_tangents_();
    }

    ~Path() {}

    const Vectors<ndim> nodes() const { return nodes_; }
    Eigen::DenseIndex num_nodes() const { return nodes().rows(); }
    Eigen::DenseIndex num_pieces() const { return num_nodes() - 1; }
    const Vectors<ndim> tangents() const { return tangents_; }
    const BdryConds<ndim> bdry_conds() const { return bdry_conds_; }
    float total_length(size_t num_params) const {
        Eigen::VectorXf params = Eigen::VectorXf::LinSpaced(num_params, 0, num_pieces());
        Eigen::VectorXf alens = arc_length_lin_approx(params);
        return alens(alens.size() - 1);
    }

    Vector<ndim> operator()(float param) const;
    Vectors<ndim> operator()(Eigen::VectorXf const &params) const;

    Eigen::MatrixXf system_matrix(bc_t bc_left, bc_t bc_right) const;
    Eigen::VectorXf system_rhs(bc_t bc_left, bc_t bc_right, size_t dim) const;
    Eigen::VectorXf arc_length_lin_approx(Eigen::VectorXf const &params) const;
    Eigen::VectorXf arc_length_params_lin_approx(size_t num_params) const;

  private:
    const Vectors<ndim> nodes_;
    const BdryConds<ndim> bdry_conds_;
    Vectors<ndim> tangents_;
    std::vector<Eigen::MatrixXf> sys_matrices_;
    std::vector<Eigen::PartialPivLU<Eigen::MatrixXf>> sys_matrix_decomps_;
    std::vector<Eigen::VectorXf> sys_rhs_;

    void compute_matrices_() {
        for (size_t dim = 0; dim < ndim; ++dim) {
            Eigen::MatrixXf sys_mat = system_matrix(bdry_conds_(dim, 0), bdry_conds_(dim, 1));
            sys_matrices_.push_back(sys_mat);
            Eigen::PartialPivLU<Eigen::MatrixXf> decomp = sys_mat.partialPivLu();
            sys_matrix_decomps_.push_back(decomp);
        }
    }

    void init_tangents_() {
        const Vector<ndim> tang_left = nodes_.row(1) - nodes_.row(0);
        const Vector<ndim> tang_right = nodes_.row(num_nodes() - 1) - nodes_.row(num_nodes() - 2);
        init_tangents_(tang_left, tang_right);
    }

    void init_tangents_(const Vector<ndim> tang_left, const Vector<ndim> tang_right) {
        tangents_.resize(num_nodes());
        tangents_.row(0) = tang_left;
        tangents_.row(num_nodes() - 1) = tang_right;
    }

    void init_rhs_() {
        for (size_t dim = 0; dim < ndim; ++dim ) {
            sys_rhs_.push_back(this->system_rhs(bdry_conds_(dim, 0), bdry_conds_(dim, 1), dim));
        }
    }

    void compute_tangents_() {
        for (size_t dim = 0; dim < ndim; ++dim) {
            tangents_.col(dim) = this->sys_matrix_decomps_[dim].solve(this->sys_rhs_[dim]);
        }
    }
}; // class Path

template <size_t ndim>
std::ostream &operator<<(std::ostream &out, Path<ndim> const &p);
