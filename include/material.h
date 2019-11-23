#ifndef RBF_MATERIAL_H
#define RBF_MATERIAL_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "types.h"

namespace rbf {

inline void compute_lame_coeffs(const double Ym, const double Pr,
                                double &mu, double &lambda) {
  mu = Ym/(2*(1+Pr));
  lambda = Ym*Pr/((1+Pr)*(1-2*Pr));
}

// set material composition from an image
class img_to_mat_converter
{
 public:
  img_to_mat_converter(const mati_t &quad, const matd_t &nods,
                       const double max_E, const double min_E)
      : quad_(quad), nods_(nods) {        
    dx_ = (nods.col(quad(0, 0))-nods.col(quad(1, 0))).norm();
    dy_ = (nods.col(quad(1, 0))-nods.col(quad(2, 0))).norm();
    
    max_lame_.setZero(2);
    compute_lame_coeffs(max_E, 0.45, max_lame_[0], max_lame_[1]);
    min_lame_.setZero(2);
    compute_lame_coeffs(min_E, 0.45, min_lame_[0], min_lame_[1]);
  }
  matd_t generate(const std::string &img_file) const {
    const int n = static_cast<int>(sqrt(quad_.cols()));
    assert(n*n == quad_.cols());

    double min_x = nods_.row(0).minCoeff(), max_x = nods_.row(0).maxCoeff();
    double min_y = nods_.row(1).minCoeff(), max_y = nods_.row(1).maxCoeff();

    cv::Mat color_img, gray_img, bin_img;
    color_img = cv::imread(img_file.c_str(), CV_LOAD_IMAGE_COLOR);
    assert(color_img.data != NULL);
    cv::resize(color_img, color_img, cv::Size(n, n), 0, 0, CV_INTER_LINEAR);
    cv::cvtColor(color_img, gray_img, CV_BGR2GRAY);
    cv::threshold(gray_img, bin_img, 127, 255, cv::THRESH_BINARY);

    matd_t mtr = matd_t::Zero(2, quad_.cols());

    #pragma omp parallel for
    for (size_t i = 0; i < quad_.cols(); ++i) {
      const vecd_t c = 0.25*(
          nods_.col(quad_(0, i))+nods_.col(quad_(1, i))+
          nods_.col(quad_(2, i))+nods_.col(quad_(3, i)));
      int q = fabs(c[0]-min_x)/dx_;
      int p = fabs(c[1]-max_y)/dy_;
      
      if ( bin_img.at<uchar>(p, q) == static_cast<uchar>(255) ) {
        mtr.col(i) = min_lame_;
      } else {
        mtr.col(i) = max_lame_;
      }
    }

    return mtr;
  }

 private:
  const mati_t &quad_;
  const matd_t &nods_;
  
  double dx_, dy_;
  vecd_t max_lame_, min_lame_;
};

}

#endif
