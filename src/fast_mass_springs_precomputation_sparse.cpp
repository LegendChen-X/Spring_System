#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
    const int n = V.rows();
    Eigen::SparseMatrix<double> Q(n,n);
    
    r = Eigen::VectorXd::Zero(E.rows());
    for(int i=0;i<E.rows();++i) r(i) = (V.row(E(i,0)) - V.row(E(i,1))).norm();
    
    std::vector<Eigen::Triplet<double> > ijvM;
    M.resize(n,n);
    for(int i=0;i<M.rows();++i) ijvM.emplace_back(i,i,m(i));
    M.setFromTriplets(ijvM.begin(),ijvM.end());
    
    signed_incidence_matrix_sparse(V.rows(),E,A);
    
    std::vector<Eigen::Triplet<double> > ijvC;
    C.resize(b.size(),n);
    for(int i=0;i<b.size();++i) ijvC.emplace_back(i,b[i],1);
    C.setFromTriplets(ijvC.begin(),ijvC.end());
    
    double mass = 1e10;
    Q = k * A.transpose() * A + 1 / (delta_t * delta_t) * M + mass * C.transpose() * C;
  /////////////////////////////////////////////////////////////////////////////
    prefactorization.compute(Q);
    return prefactorization.info() != Eigen::NumericalIssue;
}
