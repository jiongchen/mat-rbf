#ifndef RBF_MESH_H
#define RBF_MESH_H

#include <unordered_map>
#include <boost/functional/hash.hpp>
#include "types.h"

namespace rbf {

struct mesh_edge_set
{
  mesh_edge_set(const mati_t &mesh) {
    assert(mesh.rows() == 4);
    if ( !edge2idx_.empty() ) {
      edge2idx_.clear();
    }
    
    size_t cnt = 0;
    for (size_t i = 0; i < mesh.cols(); ++i) {
      for (size_t j = 0; j < mesh.rows(); ++j) {
        size_t p = mesh(j, i), q = mesh((j+1)%mesh.rows(), i);
        if ( p > q ) {
          std::swap(p, q);
        }
        const idx_pair pq = std::make_pair(p, q);
        if ( edge2idx_.find(pq) == edge2idx_.end() ) {
          edge2idx_.insert(std::make_pair(pq, cnt++));
        }
      }
    }
  }
  size_t edges_num() const {
    return edge2idx_.size();
  }
  size_t query_edge_idx(const size_t p, const size_t q) const {
    idx_pair pq = std::make_pair(p, q);
    if ( p > q ) {
      std::swap(pq.first, pq.second);
    }
    const auto it = edge2idx_.find(pq);
    return (it == edge2idx_.end() ? -1 : it->second);
  }
  std::unordered_map<idx_pair, size_t, boost::hash<idx_pair>> edge2idx_;
};

template <class MatI, class MatD>
void planar_meshgrid(const double x0, const double x1, const size_t Nx,
                     const double y0, const double y1, const size_t Ny,
                     MatI &quad, MatD &nods) {
  const double dx = (x1-x0)/Nx;
  const double dy = (y1-y0)/Ny;

  const size_t node_num = (Nx+1)*(Ny+1);
  nods.resize(2, node_num);

  //-> first x then y
  for (size_t j = 0; j < Ny+1; ++j) {
    for (size_t i = 0; i < Nx+1; ++i) {
      const size_t idx = j*(Nx+1)+i;
      nods(0, idx) = x0+i*dx;
      nods(1, idx) = y0+j*dy; 
    }
  }

  const size_t cell_num = Nx*Ny;
  quad.resize(4, cell_num);
  for (size_t j = 0; j < Ny; ++j) {
    for (size_t i = 0; i < Nx; ++i) {
      const size_t idx = j*Nx+i;
      quad(0, idx) = j*(Nx+1)+i;
      quad(1, idx) = j*(Nx+1)+i+1;
      quad(2, idx) = (j+1)*(Nx+1)+i+1;
      quad(3, idx) = (j+1)*(Nx+1)+i;      
    }
  }
}

void subdivide_quad(const mati_t &quad, const matd_t &nods,
                    mati_t &new_quad, matd_t &new_nods,
                    spmat_t *C=nullptr, spmat_t *W=nullptr,
                    const char refine_type='B') {
  mesh_edge_set pe(quad);

  const size_t new_elem_num = 4*quad.cols();
  const size_t new_nods_num = nods.cols()+pe.edges_num()+quad.cols();

  new_quad.resize(4, new_elem_num);
  new_nods.resize(2, new_nods_num);

  std::vector<trip_t> tripsC, tripsW;
  size_t curr_nods_num = 0;

  //-> corner nodes
  new_nods.leftCols(nods.cols()) = nods;
  for (size_t i = 0; i < nods.cols(); ++i) {
    tripsC.push_back(std::move(trip_t(i, i, 1.0)));
  }
  curr_nods_num += nods.cols();

  //-> edge nodes
  for (auto &entry : pe.edge2idx_) {
    size_t miss_vert_idx = curr_nods_num+entry.second-nods.cols();
    
    new_nods.col(curr_nods_num+entry.second) = 0.5*(nods.col(entry.first.first)+nods.col(entry.first.second));
    if ( refine_type == 'B' ) { //-> for Bilinear
      tripsC.push_back(std::move(trip_t(curr_nods_num+entry.second, entry.first.first, 0.5)));
      tripsC.push_back(std::move(trip_t(curr_nods_num+entry.second, entry.first.second, 0.5)));

      tripsW.push_back(std::move(trip_t(miss_vert_idx, curr_nods_num+entry.second, 1)));
      tripsW.push_back(std::move(trip_t(miss_vert_idx, entry.first.first, -0.5)));
      tripsW.push_back(std::move(trip_t(miss_vert_idx, entry.first.second, -0.5)));
    } else if ( refine_type == 'D' ) { //-> for Dirac
      tripsW.push_back(std::move(trip_t(miss_vert_idx, curr_nods_num+entry.second, 1)));
    }
  }
  curr_nods_num += pe.edges_num();

  //-> face nodes
  for (size_t i = 0; i < quad.cols(); ++i) {
    size_t miss_vert_idx = curr_nods_num+i-nods.cols();
    
    new_nods.col(curr_nods_num+i) = 0.25*
        (nods.col(quad(0, i))+nods.col(quad(1, i))+
         nods.col(quad(2, i))+nods.col(quad(3, i)));

    tripsW.push_back(std::move(trip_t(miss_vert_idx, curr_nods_num+i, 1)));
    
    for (size_t j = 0; j < 4; ++j) {
      if ( refine_type == 'B' ) {
        tripsC.push_back(std::move(trip_t(curr_nods_num+i, quad(j, i), 0.25)));
        tripsW.push_back(std::move(trip_t(miss_vert_idx, quad(j, i), -0.25)));
      } else if ( refine_type == 'D' ) {
        //-> nothing to do here
      }
    }
  }
  curr_nods_num += quad.cols();
  assert(curr_nods_num == new_nods_num);  

  if ( C != nullptr ) {
    C->resize(new_nods_num, nods.cols());
    C->setFromTriplets(tripsC.begin(), tripsC.end());
    *C = C->transpose().eval();
  }
  if ( W != nullptr ) {
    W->resize(new_nods_num-nods.cols(), new_nods_num);
    W->setFromTriplets(tripsW.begin(), tripsW.end());
  }

  //-> build connection
  const size_t edge_vert_offset = nods.cols();
  const size_t elem_vert_offset = edge_vert_offset+pe.edges_num();  
  #pragma omp parallel for
  for (size_t i = 0; i < quad.cols(); ++i) {
    const veci_t x = quad.col(i);
    
    new_quad(0, 4*i+0) = x[0];
    new_quad(1, 4*i+0) = pe.query_edge_idx(x[0], x[1])+edge_vert_offset;
    new_quad(2, 4*i+0) = i+elem_vert_offset;
    new_quad(3, 4*i+0) = pe.query_edge_idx(x[0], x[3])+edge_vert_offset;

    new_quad(0, 4*i+1) = pe.query_edge_idx(x[0], x[1])+edge_vert_offset;
    new_quad(1, 4*i+1) = x[1];
    new_quad(2, 4*i+1) = pe.query_edge_idx(x[1], x[2])+edge_vert_offset;
    new_quad(3, 4*i+1) = i+elem_vert_offset;

    new_quad(0, 4*i+2) = i+elem_vert_offset;
    new_quad(1, 4*i+2) = pe.query_edge_idx(x[1], x[2])+edge_vert_offset;
    new_quad(2, 4*i+2) = x[2];
    new_quad(3, 4*i+2) = pe.query_edge_idx(x[2], x[3])+edge_vert_offset;

    new_quad(0, 4*i+3) = pe.query_edge_idx(x[0], x[3])+edge_vert_offset;
    new_quad(1, 4*i+3) = i+elem_vert_offset;
    new_quad(2, 4*i+3) = pe.query_edge_idx(x[2], x[3])+edge_vert_offset;
    new_quad(3, 4*i+3) = x[3];
  }
}

}

#endif
