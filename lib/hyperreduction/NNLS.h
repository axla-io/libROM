/******************************************************************************
 *
 * Copyright (c) 2013-2022, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: Interface to the NNLS algorithm for sparse approximation of a
// non-negative least squares (NNLS) solution.

#ifndef included_NNLS_h
#define included_NNLS_h

#include "linalg/Matrix.h"
#include "linalg/Vector.h"

namespace CAROM {

/**
 * @brief Computes the approximate NNLS solution x for a given system matrix G,
 * tolerance tau, and RHS Gw.
 *
 * Implemented from T. Chapman et al, "Accelerated mesh sampling for the hyper
 * reduction of nonlinear computational models," Int. J. Numer. Meth. Engng.,
 * 109: 1623-1654.
 *
 * @param[in] tau The positive tolerance used for termination of iterations.
 * @param[in] G The dense system matrix.
 * @param[in] w The exact dense solution, which defines the RHS as Gw.
 * @param[out] x The sparse approximate NNLS solution.
 * @param[in] printLevel If nonzero, iteration output is written.
 */
void NNLS(const double tau,
	  Matrix const& G,
	  Vector const& w,
	  Vector & x,
	  int printLevel = 0);


/**
 * \class NNLSSolver
 * Class for solving non-negative least-squares problems.
 */
class NNLSSolver {

public:
  /**
  * Constructor*/
  NNLSSolver();

  /**
   * Destructor*/
  ~NNLSSolver();

  /**
   * Set verbosity. If set to 0: print nothing. If set to 1: just print results. If set to 2: print short update on every iteration. If set to 3: print longer update each iteration.*/
  void set_verbosity(const int verbosity_in);

  /**
   * Enumerated types of QRresidual mode. Options are 'off': the residual is calculated normally, 'on': the residual is calculated using the QR method, 'hybrid': the residual is calculated normally until we experience rounding errors, then the QR method is used. The default is 'hybrid', which should see the best performance. Recommend using 'hybrid' or 'off' only, since 'on' is computationally expensive.*/
  enum class QRresidualMode {off, on, hybrid};

  /**
   * Set the residual calculation mode for the NNLS solver. See QRresidualMode enum above for details.*/
  void set_qrresidual_mode(const QRresidualMode qr_residual_mode);

  /**
   * Solve the NNLS problem. Specifically, we find a vector soln, such that    rhs_lb < mat*soln < rhs_ub   is satisfied. mat should hold a column distributed matrix (each process has all rows, but a subset of cols). rhs_ub and rhs_lb are the true bounds divided by the number of processors, such that when the rhs_lb and rhs_ub are summed across all processes we get the true bounds. soln is a vector containing the solution. rhs_lb, rhs_ub and soln are all identical across all processes.
   The method by which we find the solution is the active-set method developed by Lawson and Hanson (1974) using scalapack functions to effect the multi-processor matrix operations. To decrease rounding errors in the case of very tight tolerances, we have the option to compute the residual using the QR factorization of A, by   res = b - Q*Q^T*b. This residual calculation results in less rounding error, but is more computationally expensive. To select whether to use the QR residual method or not, see set_qrresidual_mode above.*/
  void solve_parallel_with_scalapack(const Matrix& mat, const Vector& rhs_lb, const Vector& rhs_ub, Vector& soln);

  /**
   * Normalize the constraints such that the tolerances for each constraint (ie (UB - LB)/2 ) are equal. This seems to help the performance in most cases.*/
  void normalize_constraints(Matrix& mat, Vector& rhs_lb, Vector& rhs_ub);

  inline int getNumProcs() const { return d_num_procs; };

private:
  unsigned int n_outer_;
  unsigned int n_inner_;
  double zero_tol_;
  double const_tol_;
  int verbosity_;
  unsigned int min_nnz_; // minimum number of nonzero entries
  double res_change_termination_tol_; // threshold on relative change in residual over 100 iterations for stall sensing
  int n_proc_max_for_partial_matrix_; // maximum number of processors used in the partial matrix containing only the nonzero quadrature points.
  bool normalize_const_;
  bool QR_reduce_const_;
  bool NNLS_qrres_on_;
  QRresidualMode qr_residual_mode_;

  int d_num_procs;
  int d_rank;
};

}

#endif
