#include "fast_mass_springs_step_dense.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::MatrixXd & M,
  const Eigen::MatrixXd & A,
  const Eigen::MatrixXd & C,
  const Eigen::LLT<Eigen::MatrixXd> & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////
  // Replace with your code
    Eigen::MatrixXd p = Ucur;
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(E.rows(),3);
    const Eigen::MatrixXd l = Ucur;
    for(int iter = 0;iter<50;iter++)
    {
        for(int i=0;i<r.size();++i) d.row(i) = r(i) * (p.row(E(i,0)) - p.row(E(i,1))).normalized();
        
        Eigen::MatrixXd y = 1 / (delta_t * delta_t) * M * (2 * Ucur - Uprev) + fext;
        
        double mass = 1e10;
        
        Eigen::MatrixXd matrixB = k * A.transpose() * d + y + mass * C.transpose() * C * V;
        
        p = prefactorization.solve(matrixB);
    }
    Unext = p;
  //////////////////////////////////////////////////////////////////////////////
}
