#ifndef RBF_IO_H
#define RBF_IO_H

#include <fstream>
#include <iomanip>

#include "types.h"

namespace rbf {

template <typename OS, typename FLOAT, typename INT>
void quad2vtk(OS &os, const FLOAT *node, size_t node_num, const INT *quad, size_t quad_num) {
  os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

  os<< "POINTS " << node_num << " float\n";
  for(size_t i = 0; i < node_num; ++i)
    os << node[i*3+0] << " " << node[i*3+1] << " " << node[i*3+2] << "\n";

  os << "CELLS " << quad_num << " " << quad_num*5 << "\n";
  for(size_t i = 0; i < quad_num; ++i)
    os << 4 << "  " << quad[i*4+0] << " " << quad[i*4+1] << " " << quad[i*4+2] << " " << quad[i*4+3] << "\n";
  os << "CELL_TYPES " << quad_num << "\n";
  for(size_t i = 0; i < quad_num; ++i)
    os << 9 << "\n";
}

template <typename OS, typename Iterator, typename INT>
void vtk_data(OS &os, Iterator first, INT size, const char *value_name, const char *table_name = "my_table") {
  os << "SCALARS " << value_name << " float\nLOOKUP_TABLE " << table_name << "\n";
  for(size_t i = 0; i < size; ++i, ++first)
    os << *first << "\n";
}

//-> write quad mesh to vtk file
int write_quad_mesh(const char *path, const matd_t &nods, const mati_t &quad,
                    const matd_t *mtr=nullptr, const char *type="CELL") {
  assert(quad.rows() == 4);
  
  std::ofstream ofs(path);
  if ( ofs.fail() ) {
    std::cerr << "# cannot open " << path << std::endl;
    return __LINE__;
  }

  ofs << std::setprecision(15);

  if ( nods.rows() == 2 ) {
    matd_t tmp_nods = matd_t::Zero(3, nods.cols());
    tmp_nods.topRows(2) = nods;
    quad2vtk(ofs, tmp_nods.data(), tmp_nods.cols(), quad.data(), quad.cols());
  } else if ( nods.rows() == 3 ) {
    quad2vtk(ofs, nods.data(), nods.size()/3, quad.data(), quad.cols());    
  }

  //-> write facet-wise scalar field
  if ( mtr != nullptr ) {
    for (int i = 0; i < mtr->rows(); ++i) {
      const std::string mtr_name = "theta_"+std::to_string(i);
      const vecd_t curr_mtr = mtr->row(i);
      if ( i == 0 ) {
        ofs << type << "_DATA " << curr_mtr.size() << "\n";
      }
      vtk_data(ofs, curr_mtr.data(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
    }
  }
  ofs.close();

  return 0;
}

}

#endif
