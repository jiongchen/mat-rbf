#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <unsupported/Eigen/KroneckerProduct>

#include "energy.h"
#include "material.h"
#include "mesh.h"
#include "solver.h"
#include "io.h"

using namespace std;
using namespace Eigen;
using namespace rbf;

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# usage: ./main config.json" << endl;
    return __LINE__;
  }

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(argv[1], pt);

  const size_t level_num     = pt.get<size_t>("levels");
  const size_t res           = pt.get<size_t>("resolution");
  const char   refine_type   = pt.get<char>("refinement");
  const double min_E         = pt.get<double>("min_young");
  const double max_E         = pt.get<double>("max_young");
  const string mtr_img_file  = pt.get<string>("material");
  const double rho           = pt.get<double>("density");
  cout << "# levels in total: " << level_num << endl;
  cout << "# coarsest res: " << res << endl;
  cout << "# refinement: " << refine_type << endl;
  cout << "# density: " << rho << endl;
  cout << "# min max Young's modulus: " << min_E << " " << max_E << endl;
  cout << endl;

  const string outdir = "./result";
  boost::filesystem::create_directories(outdir);

  //-> generate mesh hierarchy
  const double X0 = 0, X1 = 1, Y0 = 0, Y1 = 1;
  std::vector<p_mra_level> levels(level_num);
  {
    for (size_t i = 0; i < levels.size(); ++i) {
      levels[i] = make_shared<mra_level>();
      auto &curr = levels[i];

      if ( i == 0 ) {
        planar_meshgrid(X0, X1, res, Y0, Y1, res, curr->cell_, curr->nods_);      
      } else {
        const auto &prev = levels[i-1];
        subdivide_quad(prev->cell_, prev->nods_, curr->cell_, curr->nods_, &curr->C_, &curr->W_, refine_type);
        curr->C_ = kroneckerProduct(curr->C_, Matrix2d::Identity()).eval();
        curr->W_ = kroneckerProduct(curr->W_, Matrix2d::Identity()).eval();
        cout << "# check CW^T=" << (curr->C_*curr->W_.transpose()).norm() << endl;
      }

      char outf[256];
      sprintf(outf, "%s/mesh-level-%02zu.vtk", outdir.c_str(), i);
      write_quad_mesh(outf, curr->nods_, curr->cell_);   
    }    
  }
  cout << endl;

  const auto &FL = levels.back();
  const auto &CL = levels.front();
  
  //-> generate material composition
  img_to_mat_converter conv(FL->cell_, FL->nods_, max_E, min_E);  
  matd_t FL_mtr = conv.generate(mtr_img_file);
  {
    const string outf = string(outdir+"/matr_setting.vtk");
    write_quad_mesh(outf.c_str(), FL->nods_, FL->cell_, &FL_mtr);
  }

  //-> get pinned node indices
  vector<size_t> FL_fix;
  {
    for (size_t i = 0; i < FL->nods_.cols(); ++i) {
      if ( fabs(FL->nods_(1, i)-Y1) < 1e-8 ) {
        FL_fix.push_back(i);
      }
    }
  }

  //-> assemble energies
  enum {ELAS, GRAV, POS};
  vector<shared_ptr<func_t>> ebf(POS+1);
  ebf[ELAS] = std::make_shared<elastic_energy>(FL->cell_, FL->nods_, FL_mtr, 1e0);
  ebf[GRAV] = std::make_shared<grav_energy>(FL->cell_,FL->nods_, rho, 1e0);
  ebf[POS]  = std::make_shared<pos_penalty>(FL->nods_, FL_fix, 1e8);
  p_func_t energy = std::make_shared<sum_func_t>(ebf);

  //-> get finest stiffness
  spmat_t FL_A(energy->dim(), energy->dim());
  {
    vector<trip_t> trips;
    energy->hes(FL->nods_.data(), &trips);
    FL_A.setFromTriplets(trips.begin(), trips.end());
  }

  //-> fine-to-coarse homogenization
  rbf_mra_solver solver(levels, pt);
  solver.coarsen(FL_A);
  {
    //-> plot the magnitude of adapted basis function for the central node
    size_t cen_idx = -1;
    double dist = 1e20;
    for (size_t i = 0; i < CL->nods_.cols(); ++i) {
      double d = (CL->nods_.col(i)-0.5*Vector2d(X0+X1, Y0+Y1)).norm();
      if ( d < dist ) {
        cen_idx = i;
        dist = d;
      }
    }   
    for (size_t i = 0; i < levels.size(); ++i) {
      matd_t bf = levels[i]->Phi_.middleRows(2*cen_idx, 2);
      matd_t mag(1, bf.cols()/2);
      for (size_t j = 0; j < mag.cols(); ++j) {
        mag(0, j) = bf.middleCols(2*j, 2).norm();
      }
      char outf[256];
      sprintf(outf, "%s/central_bf_on_level_%02zu.vtk", outdir.c_str(), i);
      write_quad_mesh(outf, FL->nods_, FL->cell_, &mag, "POINT");
    }
  }

  //-> get RHS
  vecd_t FL_g = vecd_t::Zero(energy->dim());
  {
    energy->gra(FL->nods_.data(), FL_g.data());
    FL_g *= -1;
  }

  //-> multilevel solve <FL_A*u = FL_g>
  matd_t u;  
  solver.solve(FL_g, u);
  {
    //-> write solution on each level
    assert(u.rows() == FL->nods_.size());
    matd_t x0 = FL->nods_;
    for (size_t i = 0; i < u.cols(); ++i) {
      Map<vecd_t>(x0.data(), x0.size()) += u.col(i);

      char outf[256];
      sprintf(outf, "%s/sol_on_level_%02zu.vtk", outdir.c_str(), i);
      write_quad_mesh(outf, x0, FL->cell_, &FL_mtr, "CELL");
    }
  }
  
  cout << "[INFO] done" << endl;
  return 0;
}
