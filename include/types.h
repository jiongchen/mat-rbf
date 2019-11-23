#ifndef RBF_TYPES_H
#define RBF_TYPES_H

#include <Eigen/Sparse>

namespace rbf {

typedef Eigen::MatrixXi mati_t;
typedef Eigen::MatrixXd matd_t;
typedef Eigen::VectorXi veci_t;
typedef Eigen::VectorXd vecd_t;
typedef Eigen::SparseMatrix<double> spmat_t;
typedef std::pair<size_t, size_t> idx_pair;
typedef Eigen::Triplet<double> trip_t;
typedef Eigen::SimplicialLLT<spmat_t> llt_solver_t;

}

#endif
