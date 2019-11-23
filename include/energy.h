#ifndef RBF_ENERGY_H
#define RBF_ENERGY_H

#include <cmath>
#include <Eigen/Dense>

#include "types.h"

namespace rbf {

// abstract functional
class func_t
{
 public:
  ~func_t() {}
  virtual size_t dim() const = 0;
  virtual void   val(const double *x, double *val) const = 0;
  virtual void   gra(const double *x, double *gra) const = 0;
  virtual void   hes(const double *x, std::vector<trip_t> *hes) const = 0;
};

typedef std::shared_ptr<func_t> p_func_t;

class sum_func_t : public func_t
{
 public:
  sum_func_t(const std::vector<p_func_t> &funcs)
      : buff_(funcs) {
    for (const auto &E : funcs) {
      if ( E.get() ) {
        dim_ = E->dim();
        break;
      }
    }
  }
  size_t dim() const {
    return dim_;
  }
  void val(const double *x, double *val) const {
    for (const auto &E : buff_) {
      if ( E.get() ) {
        E->val(x, val);
      }
    }
  }
  void gra(const double *x, double *gra) const {
    for (const auto &E : buff_) {
      if ( E.get() ) {
        E->gra(x, gra);
      }
    }
  }
  void hes(const double *x, std::vector<trip_t> *hes) const {
    for (const auto &E : buff_) {
      if ( E.get() ) {
        E->hes(x, hes);
      }
    }
  }

 private:
  const std::vector<p_func_t> &buff_;
  size_t dim_;
};

inline void quad_SF_jac(double *jac, const double *epsilon) {
  const double
      tt1 = 1-epsilon[1],
      tt2 = epsilon[1]+1,
      tt3 = 1-epsilon[0],
      tt4 = epsilon[0]+1;
  jac[0] = -tt1/4.0;
  jac[1] = tt1/4.0;
  jac[2] = tt2/4.0;
  jac[3] = -tt2/4.0;
  jac[4] = -tt3/4.0;
  jac[5] = -tt4/4.0;
  jac[6] = tt4/4.0;
  jac[7] = tt3/4.0;
}

#define DECLARE_VARIABLES                       \
  double X0, X1, X2, X3, X4, X5, X6, X7;        \
  if ( x ) {                                    \
    X0 = x[0]; X1 = x[1]; X2 = x[2]; X3 = x[3]; \
    X4 = x[4]; X5 = x[5], X6 = x[6], X7 = x[7]; \
  }                                             \
  double D0, D1, D2, D3, D4, D5, D6, D7;        \
  if ( d ) {                                    \
    D0 = d[0]; D1 = d[1]; D2 = d[2]; D3 = d[3]; \
    D4 = d[4]; D5 = d[5], D6 = d[6], D7 = d[7]; \
  }

#define Power(x, n) \
  std::pow(x, n)

inline void quad_elas_val(double *val, const double *x, const double *d,
                          const double *Mu, const double *Lam) {
  DECLARE_VARIABLES
  double mu = *Mu, lam = *Lam;

  *val =
      (lam*Power(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7, 2))/2.
      + mu*(Power(-1 + D0*X0 + D1*X2 + D2*X4 + D3*X6,2)
            + Power(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7,2)/2.
            + Power(-1 + D4*X1 + D5*X3 + D6*X5 + D7*X7,2));
}

inline void quad_elas_gra(double *jac, const double *x, const double *d,
                          const double *Mu, const double *Lam) {
  DECLARE_VARIABLES
  double mu = *Mu, lam = *Lam;

  double g[8] = {2*D0*mu*(-1 + D0*X0 + D1*X2 + D2*X4 + D3*X6) + 
                D4*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                D0*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                D0*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                2*D4*mu*(-1 + D4*X1 + D5*X3 + D6*X5 + D7*X7) + 
                D4*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                2*D1*mu*(-1 + D0*X0 + D1*X2 + D2*X4 + D3*X6) + 
                D5*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                D1*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                D1*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                2*D5*mu*(-1 + D4*X1 + D5*X3 + D6*X5 + D7*X7) + 
                D5*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                2*D2*mu*(-1 + D0*X0 + D1*X2 + D2*X4 + D3*X6) + 
                D6*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                D2*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                D2*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                2*D6*mu*(-1 + D4*X1 + D5*X3 + D6*X5 + D7*X7) + 
                D6*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                2*D3*mu*(-1 + D0*X0 + D1*X2 + D2*X4 + D3*X6) + 
                D7*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                D3*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7),
                D3*mu*(D4*X0 + D0*X1 + D5*X2 + D1*X3 + D6*X4 + D2*X5 + D7*X6 + D3*X7) + 
                2*D7*mu*(-1 + D4*X1 + D5*X3 + D6*X5 + D7*X7) + 
                D7*lam*(-2 + D0*X0 + D4*X1 + D1*X2 + D5*X3 + D2*X4 + D6*X5 + D3*X6 + D7*X7)};
  std::copy(g, g+8, jac);
}

inline void quad_elas_hes(double *hes, const double *x, const double *d,
                          const double *Mu, const double *Lam) {
  DECLARE_VARIABLES
  double mu = *Mu, lam = *Lam;

  double H[64] = {Power(D0,2)*lam + 2*Power(D0,2)*mu + Power(D4,2)*mu,D0*D4*lam + D0*D4*mu,
                 D0*D1*lam + 2*D0*D1*mu + D4*D5*mu,D0*D5*lam + D1*D4*mu,D0*D2*lam + 2*D0*D2*mu + D4*D6*mu,
                 D0*D6*lam + D2*D4*mu,D0*D3*lam + 2*D0*D3*mu + D4*D7*mu,D0*D7*lam + D3*D4*mu,D0*D4*lam + D0*D4*mu,
                 Power(D4,2)*lam + Power(D0,2)*mu + 2*Power(D4,2)*mu,D1*D4*lam + D0*D5*mu,
                 D4*D5*lam + D0*D1*mu + 2*D4*D5*mu,D2*D4*lam + D0*D6*mu,D4*D6*lam + D0*D2*mu + 2*D4*D6*mu,
                 D3*D4*lam + D0*D7*mu,D4*D7*lam + D0*D3*mu + 2*D4*D7*mu,D0*D1*lam + 2*D0*D1*mu + D4*D5*mu,
                 D1*D4*lam + D0*D5*mu,Power(D1,2)*lam + 2*Power(D1,2)*mu + Power(D5,2)*mu,D1*D5*lam + D1*D5*mu,
                 D1*D2*lam + 2*D1*D2*mu + D5*D6*mu,D1*D6*lam + D2*D5*mu,D1*D3*lam + 2*D1*D3*mu + D5*D7*mu,
                 D1*D7*lam + D3*D5*mu,D0*D5*lam + D1*D4*mu,D4*D5*lam + D0*D1*mu + 2*D4*D5*mu,D1*D5*lam + D1*D5*mu,
                 Power(D5,2)*lam + Power(D1,2)*mu + 2*Power(D5,2)*mu,D2*D5*lam + D1*D6*mu,
                 D5*D6*lam + D1*D2*mu + 2*D5*D6*mu,D3*D5*lam + D1*D7*mu,D5*D7*lam + D1*D3*mu + 2*D5*D7*mu,
                 D0*D2*lam + 2*D0*D2*mu + D4*D6*mu,D2*D4*lam + D0*D6*mu,D1*D2*lam + 2*D1*D2*mu + D5*D6*mu,
                 D2*D5*lam + D1*D6*mu,Power(D2,2)*lam + 2*Power(D2,2)*mu + Power(D6,2)*mu,D2*D6*lam + D2*D6*mu,
                 D2*D3*lam + 2*D2*D3*mu + D6*D7*mu,D2*D7*lam + D3*D6*mu,D0*D6*lam + D2*D4*mu,
                 D4*D6*lam + D0*D2*mu + 2*D4*D6*mu,D1*D6*lam + D2*D5*mu,D5*D6*lam + D1*D2*mu + 2*D5*D6*mu,
                 D2*D6*lam + D2*D6*mu,Power(D6,2)*lam + Power(D2,2)*mu + 2*Power(D6,2)*mu,D3*D6*lam + D2*D7*mu,
                 D6*D7*lam + D2*D3*mu + 2*D6*D7*mu,D0*D3*lam + 2*D0*D3*mu + D4*D7*mu,D3*D4*lam + D0*D7*mu,
                 D1*D3*lam + 2*D1*D3*mu + D5*D7*mu,D3*D5*lam + D1*D7*mu,D2*D3*lam + 2*D2*D3*mu + D6*D7*mu,
                 D3*D6*lam + D2*D7*mu,Power(D3,2)*lam + 2*Power(D3,2)*mu + Power(D7,2)*mu,D3*D7*lam + D3*D7*mu,
                 D0*D7*lam + D3*D4*mu,D4*D7*lam + D0*D3*mu + 2*D4*D7*mu,D1*D7*lam + D3*D5*mu,
                 D5*D7*lam + D1*D3*mu + 2*D5*D7*mu,D2*D7*lam + D3*D6*mu,D6*D7*lam + D2*D3*mu + 2*D6*D7*mu,
                 D3*D7*lam + D3*D7*mu,Power(D7,2)*lam + Power(D3,2)*mu + 2*Power(D7,2)*mu};

  std::copy(H, H+64, hes);
}

class elastic_energy : public func_t
{
 public:
  elastic_energy(const mati_t &quad, const matd_t &nods, const matd_t &lame,
                 const double w)
      : dim_(nods.size()), w_(w), lame_(lame), quad_(quad) {    
    //-> use 4 quadratures per cell
    const double qrs[2] = {-1.0/sqrt(3.0), +1.0/sqrt(3.0)};
    const double qws[2] = {1.0, 1.0};
    QUAD_NUM_ = 2*2;
    qr_.setZero(2, QUAD_NUM_);
    qw_.setOnes(QUAD_NUM_);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        const size_t idx = 2*i+j;
        qr_(0, idx) = qrs[i];
        qr_(1, idx) = qrs[j];
        qw_[idx] = qws[i]*qws[j];
      }
    }

    det_.resize(QUAD_NUM_*quad_.cols());
    Dm_.resize(8, QUAD_NUM_*quad_.cols());

    #pragma omp parallel for
    for (size_t i = 0; i < quad_.cols(); ++i) {
      matd_t X(2, 4);
      X.col(0) = nods.col(quad_(0, i));
      X.col(1) = nods.col(quad_(1, i));
      X.col(2) = nods.col(quad_(2, i));
      X.col(3) = nods.col(quad_(3, i));

      for (size_t j = 0; j < QUAD_NUM_; ++j) { // for each quadrature
        const size_t idx = QUAD_NUM_*i+j;
        matd_t H(4, 2);
        quad_SF_jac(H.data(), &qr_(0, j));

        Eigen::Matrix2d DmH = X*H;
        det_[idx] = std::fabs(DmH.determinant());
        matd_t HDmH = H*DmH.inverse();
        Dm_.col(idx) = Eigen::Map<vecd_t>(HDmH.data(), HDmH.size());
      }
    }
  }
  size_t dim() const {
    return dim_;
  }
  void val(const double *x, double *val) const {
    Eigen::Map<const matd_t> X(x, 2, dim_/2);

    double &val_cp = *val;
    
    #pragma omp parallel for reduction(+:val_cp)
    for (size_t i = 0; i < quad_.cols(); ++i) {
      matd_t vert(2, 4);
      vert.col(0) = X.col(quad_(0, i));
      vert.col(1) = X.col(quad_(1, i));
      vert.col(2) = X.col(quad_(2, i));
      vert.col(3) = X.col(quad_(3, i));      

      double value = 0;
      for (size_t j = 0; j < QUAD_NUM_; ++j) {
        const size_t idx = QUAD_NUM_*i+j;
        double vr = 0;
        quad_elas_val(&vr, vert.data(), &Dm_(0, idx), &lame_(0, i), &lame_(1, i));
        value += qw_[j]*det_[idx]*vr;       
      }

      val_cp = val_cp+w_*value;
    }
  }
  void gra(const double *x, double *gra) const {
    Eigen::Map<const matd_t> X(x, 2, dim_/2);
    Eigen::Map<matd_t> G(gra, 2, dim_/2);

    #pragma omp parallel for
    for (size_t i = 0; i < quad_.cols(); ++i) {
      matd_t vert(2, 4);
      vert.col(0) = X.col(quad_(0, i));
      vert.col(1) = X.col(quad_(1, i));
      vert.col(2) = X.col(quad_(2, i));
      vert.col(3) = X.col(quad_(3, i));      

      matd_t g = matd_t::Zero(2, 4);
      for (size_t j = 0; j < QUAD_NUM_; ++j) {
        const size_t idx = QUAD_NUM_*i+j;
        matd_t gr = matd_t::Zero(2, 4);
        quad_elas_gra(gr.data(), vert.data(), &Dm_(0, idx), &lame_(0, i), &lame_(1, i));
        g += qw_[j]*det_[idx]*gr;
      }

      #pragma omp critical
      {
        G.col(quad_(0, i)) += w_*g.col(0);
        G.col(quad_(1, i)) += w_*g.col(1);
        G.col(quad_(2, i)) += w_*g.col(2);
        G.col(quad_(3, i)) += w_*g.col(3);
      }
    }    
  }    
  void hes(const double *x, std::vector<trip_t> *hes) const {
    #pragma omp parallel for
    for (size_t i = 0; i < quad_.cols(); ++i) {

      matd_t H = matd_t::Zero(8, 8);
      for (size_t j = 0; j < QUAD_NUM_; ++j) {
        const size_t idx = QUAD_NUM_*i+j;
        matd_t Hr = matd_t::Zero(8, 8);
        quad_elas_hes(Hr.data(), nullptr, &Dm_(0, idx), &lame_(0, i), &lame_(1, i));
        H += qw_[j]*det_[idx]*Hr;
      }

      #pragma omp critical
      {
        for (size_t p = 0; p < 8; ++p) {
          for (size_t q = 0; q < 8; ++q) {
            const size_t I = 2*quad_(p/2, i)+p%2;
            const size_t J = 2*quad_(q/2, i)+q%2;
            hes->push_back(std::move(trip_t(I, J, w_*H(p, q))));
          }
        }
      }
    }
  }      

 private:
  const double w_;
  const size_t dim_;
  const mati_t &quad_;
  const matd_t &lame_;

  matd_t Dm_;
  vecd_t det_;
  matd_t qr_;
  vecd_t qw_;
  size_t QUAD_NUM_;
};

class grav_energy : public func_t
{
 public:
  grav_energy(const mati_t &quad, const matd_t &nods,
              const double rho, const double w)
      : dim_(nods.size()), w_(w), g_(9.8) {
    m_ = vecd_t::Zero(nods.cols());
    for (size_t i = 0; i < quad.cols(); ++i) {
      double a = (nods.col(quad(0, i))-nods.col(quad(1, i))).norm();
      double b = (nods.col(quad(1, i))-nods.col(quad(2, i))).norm();
      const double mass = rho*a*b;
      m_[quad(0, i)] += mass/4;
      m_[quad(1, i)] += mass/4;
      m_[quad(2, i)] += mass/4;
      m_[quad(3, i)] += mass/4;
    }
  }
  size_t dim() const {
    return dim_;
  }
  void val(const double *x, double *val) const {
    for (size_t i = 0; i < dim_/2; ++i) {
      *val += w_*m_[i]*g_*x[2*i+1];
    }
  }
  void gra(const double *x, double *gra) const {
    for (size_t i = 0; i < dim_/2; ++i) {
      gra[2*i+1] += w_*m_[i]*g_;
    }
  }
  void hes(const double *x, std::vector<trip_t> *hes) const {}

 private:
  const double w_;
  const double g_;
  const size_t dim_;
  vecd_t m_;
};

class pos_penalty : public func_t
{
 public:
  pos_penalty(const matd_t &nods, const std::vector<size_t> &idx,
              const double w)
      : x0_(nods), dim_(nods.size()), w_(w), idx_(idx) {}
  size_t dim() const {
    return dim_;
  }
  void val(const double *x, double *val) const {
    Eigen::Map<const matd_t> X(x, 2, dim_/2);

    for (const auto id : idx_) {
      *val += 0.5*w_*(X.col(id)-x0_.col(id)).squaredNorm();
    }
  }
  void gra(const double *x, double *gra) const {
    Eigen::Map<const matd_t> X(x, 2, dim_/2);
    Eigen::Map<matd_t> G(gra, 2, dim_/2);

    for (const auto id : idx_) {
      G.col(id) += w_*(X.col(id)-x0_.col(id));
    }
  }
  void hes(const double *x, std::vector<trip_t> *hes) const {
    for (const auto id : idx_) {
      hes->push_back(std::move(trip_t(2*id+0, 2*id+0, w_)));
      hes->push_back(std::move(trip_t(2*id+1, 2*id+1, w_)));
    }
  }
  
 private:
  const size_t dim_;
  const double w_;
  const std::vector<size_t> &idx_;
  const matd_t x0_;
};

}

#endif
