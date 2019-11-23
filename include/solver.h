#ifndef RBF_SOLVER_H
#define RBF_SOLVER_H

#include "types.h"

namespace rbf {

struct mra_level
{
  mati_t cell_;            // mesh connectivity
  matd_t nods_;            // mesh nodes
  
  spmat_t Psi_, Phi_;
  spmat_t A_, B_;          // stiffness 
  spmat_t C_, W_;          // refinement and its kernel
  spmat_t invC_;           // psuedo inverse of C
  spmat_t oC_;             // material adapted refinement
  vecd_t  g_, w_;          // RHS and solution

  llt_solver_t slvA_, slvB_;
};

typedef std::shared_ptr<mra_level> p_mra_level;

class rbf_mra_solver
{
 public:
  rbf_mra_solver(std::vector<p_mra_level> &levels, boost::property_tree::ptree &pt)
      : levels_(levels), pt_(pt) {}
  int coarsen(const spmat_t &A) {
    assert(A.rows() == levels_.back()->C_.cols());
    
    levels_.back()->A_ = A;
    levels_.back()->Phi_.resize(A.rows(), A.rows());
    levels_.back()->Phi_.setIdentity();

    const double trunc_tol = pt_.get<double>("trunc_tol", 1e-12);

    for (int i = levels_.size()-1; i > 0; --i) {   
      auto &up = levels_[i], &bottom = levels_[i-1];

      //-> wavelet stiffness
      bottom->B_ = up->W_*up->A_*up->W_.transpose(); 
  
      //-> compute C^{\dagger}
      llt_solver_t slv;
      spmat_t CCT = up->C_*up->C_.transpose();
      slv.compute(CCT);
      assert(slv.info() == Eigen::Success);
      up->invC_ = slv.solve(up->C_);
      assert(slv.info() == Eigen::Success);

      //-> compute ZT
      const matd_t WACT = -up->W_*up->A_*up->invC_.transpose();
      matd_t zt = matd_t::Zero(bottom->B_.cols(), WACT.cols());
      bottom->slvB_.compute(bottom->B_);
      assert(bottom->slvB_.info() == Eigen::Success);
      #pragma omp parallel for
      for (size_t k = 0; k < zt.cols(); ++k) {
        zt.col(k) = bottom->slvB_.solve(WACT.col(k));
      }
      spmat_t ZT = zt.sparseView(1, trunc_tol);
    
      //-> fine level adapated basis refinement
      matd_t oC = up->invC_+ZT.transpose()*up->W_;
      up->oC_ = oC.sparseView(1, trunc_tol); 
    
      //-> coarse level adapted basis & wavelets
      bottom->Psi_ = up->W_*up->Phi_;
      bottom->Phi_ = up->oC_*up->Phi_; 

      //-> coarse level operator by gakerkin projection
      bottom->A_ = up->oC_*up->A_*up->oC_.transpose();
      bottom->A_ = bottom->A_.pruned(1, trunc_tol).eval();  
    }

    //-> factorize last level A
    auto &last = levels_.front();
    last->slvA_.compute(last->A_);
  
    return 0;
  }
  int solve(const vecd_t &rhs, matd_t &u) const {
    //-> forward RHS
    levels_.back()->g_ = rhs;
    for (int i = levels_.size()-2; i >= 0; --i) {
      auto &curr = levels_[i], &prev = levels_[i+1];
      curr->g_ = prev->oC_*prev->g_;
    }

    u.setZero(levels_.back()->A_.cols(), levels_.size());

    //-> solve wavelet solutions
    for (int i = levels_.size()-2; i >= 0; --i) {
      auto &curr = levels_[i], &prev = levels_[i+1];
    
      const vecd_t Wg = prev->W_*prev->g_;
      curr->w_ = curr->slvB_.solve(Wg);
      assert(curr->slvB_.info() == Eigen::Success);
      u.col(i+1) = curr->Psi_.transpose()*curr->w_;
    }

    //-> solve coarsest solution
    auto &last = levels_.front();
    last->w_ = last->slvA_.solve(last->g_);
    u.col(0) = last->Phi_.transpose()*last->w_;

    return 0;
  }

 private:
  std::vector<p_mra_level> &levels_;
  boost::property_tree::ptree &pt_;
};

}

#endif
