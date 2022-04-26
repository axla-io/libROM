//               libROM MFEM Example: parametric ROM for nonlinear diffusion problem
//
// Compile with: ./scripts/compile.sh -m
//
// Description:  This example solves a time dependent nonlinear diffusion equation
//               dp/dt + div(v) = f, grad p = -a(p) v. After discretization by mixed FEM,
//               M(p) v + B^T p = 0
//               B v - C p_t = -f
//               where
//               M(p) = \int_\Omega a(p) w_h \cdot v_h d\Omega   w_h, v_h \in R_h
//               B = -\int_\Omega \div w_h q_h d\Omega   w_h \in R_h, q_h \in W_h
//               C = \int_\Omega q_h p_h d\Omega   p_h \in W_h, q_h \in W_h
//               Here, R_h is a Raviart-Thomas finite element subspace of H(div),
//               and W_h is a finite element subspace of L2.
//               The first equation allows the substitution v = -M(p)^{-1} B^T p, so
//               C p_t + B M(p)^{-1} B^T p = f
//               For the purpose of using an ODE solver, this can be expressed as
//               p_t = C^{-1} (f - B M(p)^{-1} B^T p) = F(p)

// Sample runs:
//               Analytic test (reproductive)
//               mpirun -n 1 ./mixed_nonlinear_diffusion -offline
//               mpirun -n 1 ./mixed_nonlinear_diffusion -merge -ns 1
//               mpirun -n 1 ./mixed_nonlinear_diffusion -online -rrdim 8 -rwdim 8
//
//               Initial step test (reproductive)
//               mpirun -n 1 ./mixed_nonlinear_diffusion -offline -p 1
//               mpirun -n 1 ./mixed_nonlinear_diffusion -merge -ns 1 -p 1
//               mpirun -n 1 ./mixed_nonlinear_diffusion -online -rrdim 8 -rwdim 8 -p 1
//
//               Initial step parametric test (predictive)
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -offline -id 0 -sh 0.25
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -offline -id 1 -sh 0.15
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -offline -id 2 -sh 0.35
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -merge -ns 3
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -offline -id 3 -sh 0.3
//               mpirun -n 1 ./mixed_nonlinear_diffusion -p 1 -online -rrdim 8 -rwdim 8 -sh 0.3 -id 3

#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>

#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "hyperreduction/DEIM.h"
#include "mfem/SampleMesh.hpp"


typedef enum {ANALYTIC, INIT_STEP} PROBLEM;

typedef enum {RSPACE, WSPACE} FESPACE;

//#define USE_GNAT

using namespace mfem;
using namespace std;

static bool nonlinear_problem;
static int problem;
static double diffusion_c, step_half;

class NonlinearDiffusionOperator;
class RomOperator;

// Element matrix assembly, copying element loop from BilinearForm::Assemble(int skip_zeros)
// and element matrix computation from VectorFEMassIntegrator::AssembleElementMatrix.

void AssembleElementMatrix_VectorFEMassIntegrator(Coefficient *Q,
						  const FiniteElement &el,
						  ElementTransformation &Trans,
						  DenseMatrix &elmat)
{
   int dof = el.GetDof();
   int spaceDim = Trans.GetSpaceDim();

   double w;

   DenseMatrix trial_vshape(dof, spaceDim);

   elmat.SetSize(dof);
   elmat = 0.0;

   const IntegrationRule *ir = NULL;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = Trans.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);

      el.CalcVShape(Trans, trial_vshape);

      w = ip.weight * Trans.Weight();

      if (Q)
	{
	  w *= Q -> Eval (Trans, ip);
	}
      AddMult_a_AAt (w, trial_vshape, elmat);
   }
}

SparseMatrix* Assemble_VectorFEMassIntegrator(Coefficient *Q, ParFiniteElementSpace *fes)
{
  ElementTransformation *eltrans;
  DofTransformation * doftrans;
  Mesh *mesh = fes -> GetMesh();
  DenseMatrix elmat;
  int skip_zeros = 1;

  SparseMatrix *mat = new SparseMatrix(fes->GetTrueVSize());

  Array<int> vdofs;

  for (int i = 0; i < fes -> GetNE(); i++)
    {
      doftrans = fes->GetElementVDofs(i, vdofs);
      elmat.SetSize(0);
      const FiniteElement &fe = *fes->GetFE(i);
      eltrans = fes->GetElementTransformation(i);

      //domain_integs[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
      AssembleElementMatrix_VectorFEMassIntegrator(Q, fe, *eltrans, elmat);

      if (doftrans)
	{
	  MFEM_ABORT("TODO");
	}

      mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
    }

  mat->Finalize(skip_zeros);

  return mat;
}

class NonlinearDiffusionGradientOperator : public Operator
{
private:
    friend class NonlinearDiffusionOperator;

    void SetParameters(Operator *C_solver_, Operator *M_solver_,
                       Operator *M_prime_, Operator *B_, const double dt_);

    Operator *C_solver;
    Operator *M_solver;
    Operator *M_prime;
    Operator *B;

    double dt;

    mutable Vector zR; // auxiliary vector
    mutable Vector yR; // auxiliary vector
    mutable Vector xR; // auxiliary vector
    mutable Vector zW; // auxiliary vector

public:
    NonlinearDiffusionGradientOperator(const int sizeR, const int sizeW);

    void Mult(const Vector &x, Vector &y) const;
};

NonlinearDiffusionGradientOperator::NonlinearDiffusionGradientOperator(const int sizeR,
        const int sizeW)
    : Operator(sizeW), zW(sizeW), zR(sizeR), yR(sizeR), xR(sizeR),
      C_solver(NULL), M_solver(NULL), M_prime(NULL), B(NULL), dt(0.0)
{
}

void NonlinearDiffusionGradientOperator::SetParameters(Operator *C_solver_, Operator *M_solver_,
        Operator *M_prime_, Operator *B_, const double dt_)
{
    C_solver = C_solver_;
    M_solver = M_solver_;
    M_prime = M_prime_;
    B = B_;

    dt = dt_;
}

void NonlinearDiffusionGradientOperator::Mult(const Vector &x, Vector &y) const
{
    // Gradient is I + dt C^{-1} B M(p)^{-1} B^T - dt C^{-1} B M(p)^{-1} M(a'(p)) M(p)^{-1} B^T
    //   = I + dt C^{-1} B (I - M(p)^{-1} M(a'(p))) M(p)^{-1} B^T

    // Apply M(p)^{-1} B^T
    B->MultTranspose(x, zR);
    M_solver->Mult(zR, yR);

    // Apply (I - M(p)^{-1} M(a'(p))) to yR

    M_prime->Mult(yR, zR);
    M_solver->Mult(zR, xR);

    yR.Add(-1.0, xR);

    // Apply C^{-1} B
    B->Mult(yR, zW);
    C_solver->Mult(zW, y);

    zW = y;
    y = x;
    y.Add(dt, zW);
}

class NonlinearDiffusionOperator : public TimeDependentOperator
{

protected:
    friend class RomOperator;

    Array<int> ess_Rdof_list; // this list remains empty for pure essential b.c.

    mutable ParBilinearForm *M;
    mutable HypreParMatrix *Cmat, *Mmat;
    mutable ParBilinearForm *Mprime;
    ParBilinearForm *C;

    HypreParMatrix *Bmat;
    HypreParMatrix *BTmat;
    mutable HypreParMatrix Mprimemat;

    mutable BlockOperator *fullOp;
    mutable BlockOperator *fullGradient;
    mutable BlockDiagonalPreconditioner *fullPrec;

    Array<int> block_trueOffsets;

    double current_dt;

    mutable CGSolver *M_solver;    // Krylov solver for inverting the R mass matrix M
    mutable HypreSmoother M_prec;  // Preconditioner for the R mass matrix M

    mutable CGSolver C_solver;    // Krylov solver for inverting the W mass matrix C
    HypreSmoother C_prec; // Preconditioner for the W mass matrix C

    GMRESSolver *J_gmres;

    NewtonSolver newton_solver;

    NonlinearDiffusionGradientOperator *gradient;

    double linear_solver_rel_tol;

    Vector p0;
    Vector dpdt_prev;

    mutable Vector zR; // auxiliary vector
    mutable Vector yR; // auxiliary vector
    mutable Vector zW; // auxiliary vector

public:
    NonlinearDiffusionOperator(ParFiniteElementSpace &fR, ParFiniteElementSpace &fW,
                               const double rel_tol, const double abs_tol,
                               const int iter, const Vector &p);

    virtual void Mult(const Vector &p, Vector &dp_dt) const;

    void Mult_Mmat(const Vector &p, Vector &Mp) const
    {
        Mmat->Mult(p, Mp);
    }

    void Mult_FullSystem(const Vector &p, Vector &dp_dt) const;
    void SetBTV(const CAROM::Matrix *V, CAROM::Matrix *BTV) const;

    void GetSource(Vector& s) const;

    /** Solve the Backward-Euler equation: k = f(p + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector &p, Vector &dp_dt);

    virtual Operator &GetGradient(const Vector &p) const;

    /// Update the diffusion BilinearForm K using the given true-dof vector `p`.
    void SetParameters(const Vector &p) const;

    void CopyDpDt(Vector &dpdt) const
    {
        dpdt = dpdt_prev;
    }

    void CopyDpDt_W(Vector &dpdt) const
    {
        Vector dpdt_W(dpdt_prev.GetData() + zR.Size(), zW.Size());

        dpdt = dpdt_W;
    }

    virtual ~NonlinearDiffusionOperator();

    ParFiniteElementSpace &fespace_R;  // RT finite element space
    ParFiniteElementSpace &fespace_W;  // L2 discontinuous scalar finite element space

    bool newtonFailure;
};

class RomOperator : public TimeDependentOperator
{
private:
    int rrdim, rwdim, nldim;
    int nsamp_R, nsamp_S;
    double current_dt;
    NewtonSolver newton_solver;
    GMRESSolver *J_gmres;
    CAROM::Matrix *BRsp, *BWsp;
    CAROM::Vector *psp_librom, *psp_R_librom, *psp_W_librom;
    Vector *psp;
    Vector *psp_R;
    Vector *psp_W;
    mutable Vector zR;
    mutable CAROM::Vector zY;
    mutable CAROM::Vector zN;
    const CAROM::Matrix *Vsinv;

    // Data for source function
    const CAROM::Matrix *Ssinv;
    mutable CAROM::Vector zS;
    mutable CAROM::Vector zT;
    const CAROM::Matrix *S;

    mutable DenseMatrix J;

    bool hyperreduce, hyperreduce_source;
    bool sourceFOM;

    CAROM::Vector *pfom_librom, *pfom_R_librom, *pfom_W_librom;
    Vector *pfom;
    Vector *pfom_R;
    Vector *pfom_W;
    mutable Vector zfomR;
    mutable Vector zfomW;
    CAROM::Vector *zfomR_librom;
    mutable CAROM::Vector VtzR;

    CAROM::SampleMeshManager *smm;

    void PrintFDJacobian(const Vector &p) const;

protected:
    CAROM::Matrix* BR;
    CAROM::Matrix* CR;
    const CAROM::Matrix* U_R;
    Vector y0;
    Vector dydt_prev;
    NonlinearDiffusionOperator *fom;
    NonlinearDiffusionOperator *fomSp;

public:
    RomOperator(NonlinearDiffusionOperator *fom_, NonlinearDiffusionOperator *fomSp_,
                const int rrdim_, const int rwdim_, const int nldim_,
                CAROM::SampleMeshManager *smm_,
                const CAROM::Matrix* V_R_, const CAROM::Matrix* U_R_, const CAROM::Matrix* V_W_,
                const CAROM::Matrix *Bsinv,
                const double newton_rel_tol, const double newton_abs_tol, const int newton_iter,
                const CAROM::Matrix* S_, const CAROM::Matrix *Ssinv_,
                const int myid, const bool hyperreduce_source);

    virtual void Mult(const Vector &y, Vector &dy_dt) const;
    void Mult_Hyperreduced(const Vector &y, Vector &dy_dt) const;
    void Mult_FullOrder(const Vector &y, Vector &dy_dt) const;

    /** Solve the Backward-Euler equation: k = f(p + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector &y, Vector &dy_dt);

    virtual Operator &GetGradient(const Vector &y) const;

    CAROM::Matrix V_W, V_R, VTU_R;

    CAROM::Matrix VTCS_W;

    virtual ~RomOperator();
};

// TODO: remove this by making online computation serial?
void BroadcastUndistributedRomVector(CAROM::Vector* v)
{
    const int N = v->dim();

    MFEM_VERIFY(N > 0, "");

    double *d = new double[N];

    MFEM_VERIFY(d != 0, "");

    for (int i=0; i<N; ++i)
        d[i] = (*v)(i);

    MPI_Bcast(d, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i=0; i<N; ++i)
        (*v)(i) = d[i];

    delete [] d;
}

double InitialTemperature(const Vector &x);
double SourceFunction(const Vector &x, const double t);
double ExactSolution(const Vector &x, const double t);
double NonlinearCoefficient(const double p);
double NonlinearCoefficientDerivative(const double p);

// TODO: move this to the library?
CAROM::Matrix* GetFirstColumns(const int N, const CAROM::Matrix* A)
{
    CAROM::Matrix* S = new CAROM::Matrix(A->numRows(), std::min(N, A->numColumns()), A->distributed());
    for (int i=0; i<S->numRows(); ++i)
    {
        for (int j=0; j<S->numColumns(); ++j)
            (*S)(i,j) = (*A)(i,j);
    }

    // delete A;  // TODO: find a good solution for this.
    return S;
}

// TODO: move this to the library?
void BasisGeneratorFinalSummary(CAROM::BasisGenerator* bg, const double energyFraction, int & cutoff, const std::string cutoffOutputPath)
{
    const int rom_dim = bg->getSpatialBasis()->numColumns();
    const CAROM::Vector* sing_vals = bg->getSingularValues();

    MFEM_VERIFY(rom_dim <= sing_vals->dim(), "");

    double sum = 0.0;
    for (int sv = 0; sv < sing_vals->dim(); ++sv) {
        sum += (*sing_vals)(sv);
    }

    vector<double> energy_fractions = {0.9999, 0.999, 0.99, 0.9};
    bool reached_cutoff = false;

    ofstream outfile(cutoffOutputPath);

    double partialSum = 0.0;
    for (int sv = 0; sv < sing_vals->dim(); ++sv) {
        partialSum += (*sing_vals)(sv);
        for (int i = energy_fractions.size() - 1; i >= 0; i--)
        {
            if (partialSum / sum > energy_fractions[i])
            {
                outfile << "For energy fraction: " << energy_fractions[i] << ", take first "
                        << sv+1 << " of " << sing_vals->dim() << " basis vectors" << endl;
                energy_fractions.pop_back();
            }
            else
            {
                break;
            }
        }

        if (!reached_cutoff && partialSum / sum > energyFraction)
        {
            cutoff = sv+1;
            reached_cutoff = true;
        }
    }

    if (!reached_cutoff) cutoff = sing_vals->dim();
    outfile << "Take first " << cutoff << " of " << sing_vals->dim() << " basis vectors" << endl;
    outfile.close();
}

void MergeBasis(const int dimFOM, const int nparam, const int max_num_snapshots, std::string name)
{
    MFEM_VERIFY(nparam > 0, "Must specify a positive number of parameter sets");

    bool update_right_SV = false;
    bool isIncremental = false;

    CAROM::Options options(dimFOM, nparam * max_num_snapshots, 1, update_right_SV);
    CAROM::BasisGenerator generator(options, isIncremental, "basis" + name);

    for (int paramID=0; paramID<nparam; ++paramID)
    {
        std::string snapshot_filename = "basis" + std::to_string(paramID) + "_" + name + "_snapshot";
        generator.loadSamples(snapshot_filename,"snapshot");
    }

    generator.endSamples(); // save the merged basis file

    int cutoff = 0;
    BasisGeneratorFinalSummary(&generator, 0.9999, cutoff, "mergedSV_" + name);
}

// Compute v = -M(p)^{-1} B^T p
void ComputeDualVector(ParFiniteElementSpace *fespace_R, ParFiniteElementSpace *fespace_W,
		       const HypreParMatrix *B, Vector const& p, Vector & v)
{
  MFEM_VERIFY(v.Size() == fespace_R->GetTrueVSize(), "");

  Vector rhs(fespace_R->GetTrueVSize());

  // Set grid function for a(p)
  ParGridFunction p_gf(fespace_W);

  p_gf.SetFromTrueDofs(p);

  GridFunctionCoefficient p_coeff(&p_gf);
  TransformedCoefficient a_coeff(&p_coeff, NonlinearCoefficient);
  TransformedCoefficient aprime_coeff(&p_coeff, NonlinearCoefficientDerivative);

  ParBilinearForm *M = new ParBilinearForm(fespace_R);

  M->AddDomainIntegrator(new VectorFEMassIntegrator(a_coeff));
  M->Assemble();
  M->Finalize();

  HypreParMatrix *Mmat = M->ParallelAssemble();
  CGSolver M_solver(fespace_R->GetComm());
  HypreSmoother M_prec;  // Preconditioner for the R mass matrix M
  M_prec.SetType(HypreSmoother::Jacobi);

  const double linear_solver_rel_tol = 1.0e-14;

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(linear_solver_rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(1000);
  M_solver.SetPrintLevel(-1);
  M_solver.SetPreconditioner(M_prec);

  M_solver.SetOperator(*Mmat);

  B->MultTranspose(p, rhs);
  rhs *= -1.0;
  M_solver.Mult(rhs, v);

  delete M;
  delete Mmat;
  // TODO: delete Mmat in other places?
}

void SetDualVectors(ParFiniteElementSpace *fespace_R, ParFiniteElementSpace *fespace_W,
		    const CAROM::Matrix* p, DenseMatrix & dual)
{
  ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(fespace_R, fespace_W));
  bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
  bVarf->Assemble();
  bVarf->Finalize();
  HypreParMatrix *Bmat = bVarf->ParallelAssemble();

  delete bVarf;

  Vector dual_i;
  Vector p_i(p->numRows());
  for (int i=0; i<p->numColumns(); ++i)
    {
      for (int j = 0; j < p->numRows(); ++j)
	p_i[j] = (*p)(j, i);

      dual.GetColumnReference(i, dual_i);
      ComputeDualVector(fespace_R, fespace_W, Bmat, p_i, dual_i);
    }

  delete Bmat;
}

// Compute a(p) u . v at all quadrature points on the given element. Coefficient Q is a(p);
// TODO: remove p input?
void ComputeElementRowOfG(const IntegrationRule *ir, Array<int> const& vdofs,
			  Coefficient *Q, Vector const& p, Vector const& u, Vector const& v,
			  FiniteElement const& fe, ElementTransformation & Trans, Vector & r)
{
  MFEM_VERIFY(r.Size() == ir->GetNPoints(), "");
  int dof = fe.GetDof();
  int spaceDim = Trans.GetSpaceDim();

  Vector u_i(spaceDim);
  Vector v_i(spaceDim);

  DenseMatrix trial_vshape(dof, spaceDim);

  for (int i = 0; i < ir->GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);

      fe.CalcVShape(Trans, trial_vshape);

      double w = ip.weight * Trans.Weight();

      u_i = 0.0;
      v_i = 0.0;

      for (int j=0; j<dof; ++j)
	for (int k=0; k<spaceDim; ++k)
	  {
	    u_i[k] += u[vdofs[j]] * trial_vshape(j, k);
	    v_i[k] += v[vdofs[j]] * trial_vshape(j, k);
	  }

      if (Q)
	{
	  w *= Q -> Eval (Trans, ip);
	}

      r[i] = 0.0;
      for (int k=0; k<spaceDim; ++k)
	{
	  r[i] += u_i[k] * v_i[k];
	}

      r[i] *= w;
   }
}

void NNLS(DenseMatrix const& Gt, const int ne, Array<double> const& w_el, Vector & x)
{
  const double tau = 1.0e-2;  // TODO: input this

  const int m = Gt.NumRows();
  const int n = Gt.NumCols();

  const int nqe = w_el.Size();

  MFEM_VERIFY(m == ne * nqe, "");

  Vector b(n);
  Vector w(m);
  Vector r(n);
  Vector u(m);

  Vector z(m);
  x.SetSize(m);

  CAROM::Vector bc(b.GetData(), n, false, false);

  for (int i=0; i<ne; ++i)
    {
      for (int j=0; j<nqe; ++j)
	w[(i*nqe) + j] = w_el[j];
    }

  Gt.MultTranspose(w, b);  // b = Gw

  r = b;
  x = 0.0;

  const double bnorm = b.Norml2();

  std::set<int> ids;

  int outer = 0;
  while (r.Norml2() >= tau * bnorm)
    {
      cout << "Outer loop iter " << outer << ", r norm " << r.Norml2() << endl;
      outer++;

      Gt.Mult(r, u);

      // Set id to max_value_index(u), without absolute values.
      int id = 0;
      double maxu = u[0];
      for (int i=1; i<m; ++i)
	{
	  const double u_i = u[i];
	  if (u_i > maxu)
	    {
	      maxu = u_i;
	      id = i;
	    }
	}

      ids.insert(id);

      int inner = 0;
      while (true)
	{
	  cout << "Inner loop iter " << inner << endl;
	  inner++;

	  // Extract the submatrix G_{ids}
	  CAROM::Matrix Gsub(n, ids.size(), false);

	  int count = 0;
	  for (auto i : ids)
	    {
	      for (int j=0; j<n; ++j)
		{
		  Gsub(j, count) = Gt(i, j);
		}
	      count++;
	    }

	  MFEM_VERIFY(Gsub.numColumns() < n, "");

	  // Compute the pseudo-inverse of Gsub, storing its transpose in Gsub.
	  Gsub.transposePseudoinverse();

	  CAROM::Vector t(Gsub.numColumns(), false);
	  Gsub.transposeMult(bc, t);

	  z = 0.0;

	  double minz = t(0);
	  count = 0;
	  for (auto i : ids)
	    {
	      z[i] = t(count);

	      if (t(count) < minz)
		minz = t(count);

	      count++;
	    }

	  if (minz > 0.0)
	    {
	      x = z;
	      break;
	    }

	  // Find max feasible step from x to z, which is at most 1.
	  double step = 1.0;
	  bool initStep = false;
	  for (int i=0; i<m; ++i)
	    {
	      // Solve for x + s * (z - x) = 0 => s = x / (x - z), assuming x != z.
	      if (x[i] != 0.0 && fabs(x[i] - z[i]) > 1.0e-8 && z[i] <= 0.0)
		{
		  const double s = x[i] / (x[i] - z[i]);
		  if (initStep)
		    {
		      step = std::min(step, s);
		    }
		  else
		    {
		      initStep = true;
		      step = s;
		    }
		}
	    }

	  // Set x = x + step (z - x) = (1 - step) x + step z
	  x *= 1.0 - step;
	  x.Add(step, z);

	  // Find the zero value indices of x and set ids to {1, ..., m} \ {zero value indices}
	  //ids.clear();
	  for (int i=0; i<m; ++i)  // TODO: can you just loop over entries in ids?
	    {
	      //zero[i] = (x[i] <= 0.0);
	      /*
	      if (x[i] > 0.0)
		{
		  ids.insert(i);
		}
	      */
	      if (x[i] <= 0.0)
		{
		  ids.erase(i);
		}
	    }
	} // inner loop

      // Update r = b - Gx
      r = b;
      Gt.AddMultTranspose_a(-1.0, x, r);
    }  // outer loop
}

void SetupEQP(ParFiniteElementSpace *fespace_R, ParFiniteElementSpace *fespace_W, const CAROM::Matrix* BR, const CAROM::Matrix* BW, Vector & sol)
{
  const IntegrationRule *ir0 = NULL;
  if (ir0 == NULL)
    {
      // int order = 2 * el.GetOrder();
      const FiniteElement &fe = *fespace_R->GetFE(0);
      ElementTransformation *eltrans = fespace_R->GetElementTransformation(0);

      int order = eltrans->OrderW() + 2 * fe.GetOrder();
      ir0 = &IntRules.Get(fe.GetGeomType(), order);
    }

  const int nqe = ir0->GetNPoints();
  const int ne = fespace_R->GetNE();
  const int NB = BR->numColumns();
  const int NQ = ne * nqe;

  MFEM_VERIFY(NB == BW->numColumns(), "TODO: generalize this");

  // Compute R vectors dual to W basis vectors
  DenseMatrix Rdual(BR->numRows(), NB);
  SetDualVectors(fespace_R, fespace_W, BW, Rdual);

  MFEM_VERIFY(NB == Rdual.Width(), "");

  // Compute G of size NB^2 x NQ

  DenseMatrix Gt(NQ, NB*NB);  // Store Gt rather than G, since DenseMatrix is traversed column-wise, and NQ >> NB^2.

  Vector Rdual_i;
  Vector p_i(BW->numRows());
  Vector v_j(BR->numRows());

  Vector r(nqe);

  for (int i=0; i<NB; ++i)
    {
      for (int j = 0; j < BW->numRows(); ++j)
	p_i[j] = (*BW)(j, i);

      // Set grid function for a(p)
      ParGridFunction p_gf(fespace_W);

      p_gf.SetFromTrueDofs(p_i);

      GridFunctionCoefficient p_coeff(&p_gf);
      TransformedCoefficient a_coeff(&p_coeff, NonlinearCoefficient);

      Rdual.GetColumnReference(i, Rdual_i);

      for (int j=0; j<NB; ++j)
	{
	  for (int k = 0; k < BR->numRows(); ++k)
	    v_j[k] = (*BR)(k, j);

	  // TODO: is it better to make the element loop the outer loop?
	  for (int e=0; e<ne; ++e)
	    {
	      Array<int> vdofs;
	      DofTransformation *doftrans = fespace_R->GetElementVDofs(e, vdofs);
	      const FiniteElement &fe = *fespace_R->GetFE(i);
	      ElementTransformation *eltrans = fespace_R->GetElementTransformation(i);

	      ComputeElementRowOfG(ir0, vdofs, &a_coeff, p_i, Rdual_i, v_j, fe, *eltrans, r);

	      for (int m=0; m<nqe; ++m)
		Gt((e*nqe) + m, j + (i*NB)) = r[m];
	    }
	}
    }

  Array<double> const& w_el = ir0->GetWeights();
  MFEM_VERIFY(w_el.Size() == nqe, "");

  NNLS(Gt, ne, w_el, sol);
  int nnz = 0;
  for (int i=0; i<sol.Size(); ++i)
    {
      if (sol[i] != 0.0)
	nnz++;
    }

  cout << "Number of nonzeros in NNLS solution: " << nnz << ", out of " << sol.Size() << endl;
}

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    nonlinear_problem = true;
    problem = ANALYTIC;
    diffusion_c = 2.0;
    step_half = 0.25;
    const char *mesh_file = "../data/inline-quad.mesh";
    int ser_ref_levels = 2;
    int par_ref_levels = 1;
    int order = 0;
    double t_final = 1.0e-1;
    double maxdt = 1.0e-3;
    double dt = maxdt;
    bool visualization = false;
    bool visit = false;
    int vis_steps = 5;
    double newton_rel_tol = 1e-4;
    double newton_abs_tol = 1e-10;
    int newton_iter = 10;

    // ROM parameters
    bool offline = false;
    bool merge = false;
    bool online = false;

    int nsets = 0;

    int id_param = 0;

    int rdim = -1;  // number of basis vectors to use
    int rrdim = -1;  // number of basis vectors to use
    int rwdim = -1;  // number of basis vectors to use

    int precision = 16;
    cout.precision(precision);

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Order (degree) of the finite elements.");
    args.AddOption(&id_param, "-id", "--id", "Parametric index");
    args.AddOption(&problem, "-p", "--problem", "Problem setup to use");
    args.AddOption(&step_half, "-sh", "--stephalf", "Initial step function half-width");
    args.AddOption(&diffusion_c, "-dc", "--diffusion-constant", "Diffusion coefficient constant term");
    args.AddOption(&rrdim, "-rrdim", "--rrdim",
                   "Basis dimension for H(div) vector finite element space.");
    args.AddOption(&rwdim, "-rwdim", "--rwdim",
                   "Basis dimension for L2 scalar finite element space.");
    args.AddOption(&t_final, "-tf", "--t-final",
                   "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                   "Time step.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    args.AddOption(&nsets, "-ns", "--nset", "Number of parametric snapshot sets");
    args.AddOption(&offline, "-offline", "--offline", "-no-offline", "--no-offline",
                   "Enable or disable the offline phase.");
    args.AddOption(&online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable the online phase.");
    args.AddOption(&merge, "-merge", "--merge", "-no-merge", "--no-merge",
                   "Enable or disable the merge phase.");

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }

    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    const bool check = (offline && !merge && !online) || (!offline && merge && !online) || (!offline && !merge && online);
    MFEM_VERIFY(check, "only one of offline, merge, or online must be true!");

    const bool hyperreduce_source = (problem != INIT_STEP);

    StopWatch solveTimer, totalTimer;
    totalTimer.Start();

    // 3. Read the serial mesh from the given mesh file on all processors. We can
    //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    const int dim = mesh->Dimension();

    // 4. Define the ODE solver used for time integration. For this example,
    //    only backward Euler is currently supported.
    BackwardEulerSolver ode_solver;

    // 5. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }

    // 7. Define the mixed finite element spaces.

    RT_FECollection hdiv_coll(order, dim);
    L2_FECollection l2_coll(order, dim);

    ParFiniteElementSpace R_space(pmesh, &hdiv_coll);
    ParFiniteElementSpace W_space(pmesh, &l2_coll);

    HYPRE_Int dimW = W_space.GlobalTrueVSize();
    HYPRE_Int dimR = R_space.GlobalTrueVSize();

    if (myid == 0)
    {
        cout << "Global number of L2 unknowns: " << dimW << endl;
        cout << "Global number of RT unknowns: " << dimR << endl;
    }

    bool update_right_SV = false;
    bool isIncremental = false;
    const std::string basisFileName = "basis" + std::to_string(id_param);
    int max_num_snapshots = int(t_final/dt) + 2;

    // The merge phase
    if (merge)
    {
        totalTimer.Clear();
        totalTimer.Start();

        MergeBasis(R_space.GetTrueVSize(), nsets, max_num_snapshots, "R");
        MergeBasis(R_space.GetTrueVSize(), nsets, max_num_snapshots, "FR");
        MergeBasis(W_space.GetTrueVSize(), nsets, max_num_snapshots, "W");

        if (hyperreduce_source)
        {
            MergeBasis(W_space.GetTrueVSize(), nsets, max_num_snapshots, "S");
        }

        totalTimer.Stop();
        if (myid == 0)
        {
            printf("Elapsed time for merging and building ROM basis: %e second\n", totalTimer.RealTime());
        }
        MPI_Finalize();
        return 0;
    }

    ParGridFunction p_gf(&W_space);

    // 8. Set the initial conditions for p.

    FunctionCoefficient p_0(InitialTemperature);
    p_gf.ProjectCoefficient(p_0);
    Vector p, pprev, dpdt, source;
    Vector *p_W = &p;
    ParGridFunction* sp_p_gf = 0;
    Vector sp_p;
    Vector *sp_p_W = &sp_p;

    Vector *wMFEM = 0;

    CAROM::Vector *p_librom = 0;
    CAROM::Vector *p_W_librom = 0;
    CAROM::Vector* w_W = 0;
    CAROM::Vector* w = 0;

    const int N1 = R_space.GetTrueVSize();
    const int N2 = W_space.GetTrueVSize();
    const int fdim = N1 + N2;

    cout << myid << ": Local number of L2 unknowns: " << N2 << endl;
    cout << myid << ": Local number of RT unknowns: " << N1 << endl;

    {
        p_librom = new CAROM::Vector(fdim, true);
        p.SetDataAndSize(&((*p_librom)(0)), fdim);
        p_W_librom = new CAROM::Vector(&((*p_librom)(N1)), N2, true, false);

        p = 0.0;
        p_W = new Vector(p.GetData() + N1, N2);
        p_gf.GetTrueDofs(*p_W);

        source.SetSize(N2);
    }

    // 9. Initialize the diffusion operator and the VisIt visualization.
    NonlinearDiffusionOperator oper(R_space, W_space, newton_rel_tol, newton_abs_tol, newton_iter, p);  // FOM operator
    NonlinearDiffusionOperator *soper = 0;  // Sample mesh operator

    if (offline)
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "nldiff-mesh." << setfill('0') << setw(6) << myid;
        sol_name << "nldiff-init" << id_param << "." << setfill('0') << setw(6) << myid;
        ofstream omesh(mesh_name.str().c_str());
        omesh.precision(precision);
        pmesh->Print(omesh);
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        p_gf.Save(osol);
    }

    VisItDataCollection * visit_dc = NULL;
    if (visit)
    {
        if (offline)
            visit_dc = new VisItDataCollection("nldiff-fom", pmesh);
        else
            visit_dc = new VisItDataCollection("nldiff-rom", pmesh);

        visit_dc->RegisterField("temperature", &p_gf);
        visit_dc->SetCycle(0);
        visit_dc->SetTime(0.0);
        visit_dc->Save();
    }

    socketstream sout;
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        sout.open(vishost, visport);
        sout << "parallel " << num_procs << " " << myid << endl;
        int good = sout.good(), all_good;
        MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
        if (!all_good)
        {
            sout.close();
            visualization = false;
            if (myid == 0)
            {
                cout << "Unable to connect to GLVis server at "
                     << vishost << ':' << visport << endl;
                cout << "GLVis visualization disabled.\n";
            }
        }
        else
        {
            sout.precision(precision);
            sout << "solution\n" << *pmesh << p_gf;
            sout << "pause\n";
            sout << flush;
            if (myid == 0)
            {
                cout << "GLVis visualization paused."
                     << " Press space (in the GLVis window) to resume it.\n";
            }
        }
    }

    CAROM::BasisGenerator *basis_generator_R = 0;  // For the solution component in vector H(div)
    CAROM::BasisGenerator *basis_generator_W = 0;  // For the solution component in scalar L2
    CAROM::BasisGenerator *basis_generator_FR = 0; // For the nonlinear term M(p)v with p in L2, v in H(div)
    CAROM::BasisGenerator *basis_generator_S = 0;  // For the source in scalar L2

    if (offline) {
        CAROM::Options options_R(R_space.GetTrueVSize(), max_num_snapshots, 1, update_right_SV);
        CAROM::Options options_W(W_space.GetTrueVSize(), max_num_snapshots, 1, update_right_SV);

        if (hyperreduce_source)
            basis_generator_S = new CAROM::BasisGenerator(options_W, isIncremental, basisFileName + "_S");

        basis_generator_R = new CAROM::BasisGenerator(options_R, isIncremental, basisFileName + "_R");
        basis_generator_W = new CAROM::BasisGenerator(options_W, isIncremental, basisFileName + "_W");

        basis_generator_FR = new CAROM::BasisGenerator(options_R, isIncremental, basisFileName + "_FR");
    }

    RomOperator *romop = 0;

    const CAROM::Matrix* B_librom = 0;
    const CAROM::Matrix* BR_librom = 0;
    const CAROM::Matrix* FR_librom = 0;
    const CAROM::Matrix* BW_librom = 0;
    const CAROM::Matrix* S_librom = 0;

    int nsamp_R = -1;
    int nsamp_S = -1;

    CAROM::SampleMeshManager *smm = nullptr;

    if (online)
    {
        CAROM::BasisReader readerR("basisR");
        BR_librom = readerR.getSpatialBasis(0.0);
        if (rrdim == -1)
            rrdim = BR_librom->numColumns();
        else
            BR_librom = GetFirstColumns(rrdim, BR_librom);  // TODO: reduce rrdim if too large

        MFEM_VERIFY(BR_librom->numRows() == N1, "");

        if (myid == 0)
            printf("reduced R dim = %d\n",rrdim);

        CAROM::BasisReader readerW("basisW");
        BW_librom = readerW.getSpatialBasis(0.0);
        if (rwdim == -1)
            rwdim = BW_librom->numColumns();
        else
            BW_librom = GetFirstColumns(rwdim, BW_librom);

        MFEM_VERIFY(BW_librom->numRows() == N2, "");

        if (myid == 0)
            printf("reduced W dim = %d\n",rwdim);

        // TODO: To get the basis U_R, considering the relation M(p) v + B^T p = 0 in the FOM, we can just use
        // U_R = B^T V_W. Note that V_W and V_R may have different numbers of columns, which is fine.
        // TODO: maybe we need POD applied to B^T multiplied by W-snapshots, or just a basis generator for
        // snapshots of M(p) v. This could be different from B^T multiplied by POD results for the W solutions.

        /*
        FR_librom = new CAROM::Matrix(N1, rwdim, true);
        oper.SetBTV(BW_librom, FR_librom);
        */

        CAROM::BasisReader readerFR("basisFR");
        FR_librom = readerFR.getSpatialBasis(0.0);

        // Compute sample points using DEIM, for hyperreduction

        // TODO: reduce this?
        const int nldim = FR_librom->numColumns(); // rwdim;

        cout << "FR dim " << FR_librom->numColumns() << endl;

        MFEM_VERIFY(FR_librom->numRows() == N1 && FR_librom->numColumns() >= nldim, "");

        if (FR_librom->numColumns() > nldim)
            FR_librom = GetFirstColumns(nldim, FR_librom);

	// TODO: make the option to do EQP or GNAT. Currently it does both.

	// EQP setup
	Vector eqpSol;
	SetupEQP(&R_space, &W_space, BR_librom, BW_librom, eqpSol);

	// GNAT setup
        vector<int> num_sample_dofs_per_proc(num_procs);

        nsamp_R = nldim;

#ifdef USE_GNAT
        vector<int> sample_dofs(nsamp_R);  // Indices of the sampled rows
        CAROM::Matrix *Bsinv = new CAROM::Matrix(nsamp_R, nldim, false);
        CAROM::GNAT(FR_librom,
                    nldim,
                    sample_dofs,
                    num_sample_dofs_per_proc,
                    *Bsinv,
                    myid,
                    num_procs,
                    nsamp_R);
#else
        // Now execute the DEIM algorithm to get the sampling information.
        CAROM::Matrix *Bsinv = new CAROM::Matrix(nldim, nldim, false);
        vector<int> sample_dofs(nldim);  // Indices of the sampled rows
        CAROM::DEIM(FR_librom,
                    nldim,
                    sample_dofs,
                    num_sample_dofs_per_proc,
                    *Bsinv,
                    myid,
                    num_procs);
#endif

        vector<int> sample_dofs_S;  // Indices of the sampled rows
        vector<int> num_sample_dofs_per_proc_S(num_procs);

        vector<int> sample_dofs_withS;  // Indices of the sampled rows
        int nsdim = 0;
        CAROM::Matrix *Ssinv = 0;
        vector<int> num_sample_dofs_per_proc_withS;
        CAROM::BasisReader *readerS = 0;
        if (hyperreduce_source)
        {
            readerS = new CAROM::BasisReader("basisS");
            S_librom = readerS->getSpatialBasis(0.0);

            // Compute sample points using DEIM

            nsdim = S_librom->numColumns();

            cout << "S dim " << nsdim << endl;

            // Now execute the DEIM algorithm to get the sampling information.
            nsamp_S = nsdim;
            sample_dofs_S.resize(nsamp_S);

#ifdef USE_GNAT
            Ssinv = new CAROM::Matrix(nsamp_S, nsdim, false);
            sample_dofs_S.resize(nsamp_S);

            CAROM::GNAT(S_librom,
                        nsdim,
                        sample_dofs_S,
                        num_sample_dofs_per_proc_S,
                        *Ssinv,
                        myid,
                        num_procs,
                        nsamp_S);
#else
            Ssinv = new CAROM::Matrix(nsdim, nsdim, false);
            sample_dofs_S.resize(nsdim);
            CAROM::DEIM(S_librom,
                        nsdim,
                        sample_dofs_S,
                        num_sample_dofs_per_proc_S,
                        *Ssinv,
                        myid,
                        num_procs);
#endif
        }

        // Construct sample mesh

        const int nspaces = 2;
        std::vector<ParFiniteElementSpace*> fespace(nspaces);
        std::vector<ParFiniteElementSpace*> spfespace(nspaces);
        fespace[0] = &R_space;
        fespace[1] = &W_space;

        ParFiniteElementSpace *sp_R_space, *sp_W_space;

        smm = new CAROM::SampleMeshManager(fespace);

        vector<int> sample_dofs_empty;  // Potential variable in W space has no sample DOFs.
        vector<int> num_sample_dofs_per_proc_empty;
        num_sample_dofs_per_proc_empty.assign(num_procs, 0);
        smm->RegisterSampledVariable("P", WSPACE, sample_dofs_empty, num_sample_dofs_per_proc_empty);

        if (hyperreduce_source)
        {
            smm->RegisterSampledVariable("V", RSPACE, sample_dofs, num_sample_dofs_per_proc);
            smm->RegisterSampledVariable("S", WSPACE, sample_dofs_S, num_sample_dofs_per_proc_S);
        }
        else
        {
            smm->RegisterSampledVariable("V", RSPACE, sample_dofs, num_sample_dofs_per_proc);
        }

        smm->ConstructSampleMesh();

        w = new CAROM::Vector(rrdim + rwdim, false);
        w_W = new CAROM::Vector(rwdim, false);

        // Initialize w = B_W^T p.
        BW_librom->transposeMult(*p_W_librom, *w_W);

        for (int i=0; i<rrdim; ++i)
            (*w)(i) = 0.0;

        for (int i=0; i<rwdim; ++i)
            (*w)(rrdim + i) = (*w_W)(i);

        // Note that some of this could be done only on the ROM solver process, but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = new Vector(&((*w)(0)), rrdim + rwdim);

        if (myid == 0)
        {
            sp_R_space = smm->GetSampleFESpace(RSPACE);
            sp_W_space = smm->GetSampleFESpace(WSPACE);

            // Initialize sp_p with initial conditions.
            {
                sp_p_gf = new ParGridFunction(sp_W_space);
                sp_p_gf->ProjectCoefficient(p_0);

                sp_p.SetSize(sp_R_space->GetTrueVSize() + sp_W_space->GetTrueVSize());
                sp_p = 0.0;
                sp_p_W = new Vector(sp_p.GetData() + sp_R_space->GetTrueVSize(), sp_W_space->GetTrueVSize());
                sp_p_gf->GetTrueDofs(*sp_p_W);
            }

            soper = new NonlinearDiffusionOperator(*sp_R_space, *sp_W_space, newton_rel_tol, newton_abs_tol, newton_iter, sp_p);
        }

        romop = new RomOperator(&oper, soper, rrdim, rwdim, nldim, smm,
                                BR_librom, FR_librom, BW_librom,
                                Bsinv, newton_rel_tol, newton_abs_tol, newton_iter,
                                S_librom, Ssinv, myid, hyperreduce_source);

        ode_solver.Init(*romop);

        delete readerS;
    }
    else  // fom
        ode_solver.Init(oper);

    // 10. Perform time-integration (looping over the time iterations, ti, with a time-step dt).
    double t = 0.0;
    int lastStepChange = 1;

    ConstantCoefficient coeff0(0.0);

    oper.newtonFailure = false;

    solveTimer.Start();

    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
        if (offline && !oper.newtonFailure)
        {
            const bool sampleW = basis_generator_W->isNextSample(t);

            if (sampleW && hyperreduce_source) // TODO: Instead, basis_generator_S->isNextSample(t) could be used if dS/dt were computed.
            {
                oper.GetSource(source);
                basis_generator_S->takeSample(source.GetData(), t, dt);
                // TODO: dfdt? In this example, one can implement the exact formula.
                //   In general, one can use finite differences in time (dpdt is computed that way).
                //basis_generator_S->computeNextSampleTime(p.GetData(), dfdt.GetData(), t);
            }

            if (basis_generator_R->isNextSample(t))
            {
                oper.CopyDpDt(dpdt);

                basis_generator_R->takeSample(p.GetData(), t, dt);
                basis_generator_R->computeNextSampleTime(p.GetData(), dpdt.GetData(), t);

                Vector p_R(p.GetData(), N1);
                Vector Mp(N1);
                oper.SetParameters(p);
                oper.Mult_Mmat(p_R, Mp);
                basis_generator_FR->takeSample(Mp.GetData(), t, dt);
            }

            if (sampleW)
            {
                oper.CopyDpDt_W(dpdt);

                basis_generator_W->takeSample(p_W->GetData(), t, dt);
                basis_generator_W->computeNextSampleTime(p_W->GetData(), dpdt.GetData(), t);
            }
        }

        if (online)
        {
            if (myid == 0)
            {
                ode_solver.Step(*wMFEM, t, dt);
            }

            MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else  // fom
        {
            oper.newtonFailure = false;
            pprev = p;  // Save solution, to reset in case of a Newton failure.
            const double tprev = t;
            ode_solver.Step(p, t, dt);

            if (oper.newtonFailure)
            {
                // Reset and retry.
                p = pprev;
                t = tprev;
                dt *= 0.5;
                cout << "step " << ti << ", t = " << t << " had a Newton failure, cutting dt to " << dt << endl;
                lastStepChange = ti;
                ti--;
                continue;
            }
            else if (ti - lastStepChange > 10)
            {
                dt *= 2.0;
                dt = min(dt, maxdt);
                lastStepChange = ti;
            }
        }

        if (myid == 0)
        {
            cout << "step " << ti << ", t = " << t << " == " << oper.GetTime() << ", dt " << dt << endl;

            if (online)
                cout << "rom time " << romop->GetTime() << endl;
        }

        if (t >= t_final - dt/2)
            last_step = true;

        if (last_step || (ti % vis_steps) == 0)
        {
            FunctionCoefficient exsol(ExactSolution);
            exsol.SetTime(oper.GetTime());

            if (online)  // Lift ROM solution to FOM.
            {
                exsol.SetTime(t);

                BroadcastUndistributedRomVector(w);

                for (int i=0; i<rwdim; ++i)
                    (*w_W)(i) = (*w)(rrdim + i);

                romop->V_W.mult(*w_W, *p_W_librom);

                p_gf.SetFromTrueDofs(*p_W);

                if (last_step)
                {
                    // Calculate the relative l2 error between the final ROM solution and FOM solution, using id_param for FOM solution.
                    Vector fom_solution(N2);
                    ifstream solution_file;
                    ostringstream solution_filename, rom_filename;
                    solution_filename << "nldiff-fom-values-final" << id_param << "." << setfill('0') << setw(6) << myid;
                    rom_filename << "nldiff-rom-final" << id_param << "." << setfill('0') << setw(6) << myid;

                    if (myid == 0) std::cout << "Comparing current run to solution at: " << solution_filename.str() << " with offline parameter index " << id_param << std::endl;
                    solution_file.open(solution_filename.str());
                    fom_solution.Load(solution_file, N2);
                    solution_file.close();
                    const double fomNorm = sqrt(InnerProduct(MPI_COMM_WORLD, fom_solution, fom_solution));
                    //const double romNorm = sqrt(InnerProduct(MPI_COMM_WORLD, *p_W, *p_W));
                    fom_solution -= *p_W;
                    const double diffNorm = sqrt(InnerProduct(MPI_COMM_WORLD, fom_solution, fom_solution));
                    if (myid == 0) std::cout << "Relative l2 error of ROM solution " << diffNorm / fomNorm << std::endl;

                    ofstream osol(rom_filename.str().c_str());
                    osol.precision(precision);
                    p_gf.Save(osol);
                }
            }
            else
                p_gf.SetFromTrueDofs(*p_W);

            if (problem == ANALYTIC)
            {
                const double l2err = p_gf.ComputeL2Error(exsol);
                const double l2nrm = p_gf.ComputeL2Error(coeff0);

                if (myid == 0)
                    cout << "L2 norm of exact error: " << l2err << ", FEM solution norm " << l2nrm << ", relative norm " << l2err / l2nrm << endl;
            }

            if (visualization)
            {
                sout << "parallel " << num_procs << " " << myid << "\n";
                sout << "solution\n" << *pmesh << p_gf << flush;
            }

            if (visit)
            {
                visit_dc->SetCycle(ti);
                visit_dc->SetTime(t);
                visit_dc->Save();
            }
        }
    }  // timestep loop

    solveTimer.Stop();
    if (myid == 0) cout << "Elapsed time for time integration loop " << solveTimer.RealTime() << endl;

    if (visit)
        delete visit_dc;

    if (offline)
    {
        // Sample final solution, to prevent extrapolation in ROM between the last sample and the end of the simulation.

        oper.CopyDpDt(dpdt);

        // R space
        basis_generator_R->takeSample(p.GetData(), t, dt);

        Vector p_R(p.GetData(), N1);
        Vector Mp(N1);
        oper.SetParameters(p);
        oper.Mult_Mmat(p_R, Mp);
        basis_generator_FR->takeSample(Mp.GetData(), t, dt);

        // Terminate the sampling and write out information.
        basis_generator_R->writeSnapshot();
        basis_generator_FR->writeSnapshot();

        // W space

        // TODO: why call computeNextSampleTime if you just do takeSample on every step anyway?
        basis_generator_W->takeSample(p_W->GetData(), t, dt);
        basis_generator_W->writeSnapshot();

        oper.GetSource(source);

        if (hyperreduce_source)
        {
            basis_generator_S->takeSample(source.GetData(), t, dt);
            basis_generator_S->writeSnapshot();
        }

        delete basis_generator_R;
        delete basis_generator_FR;
        delete basis_generator_W;
        delete basis_generator_S;
    }

    // 11. Save the final solution in parallel. This output can be viewed later
    //     using GLVis: "glvis -np <np> -m nldiff-mesh -g nldiff-final".
    if (offline)
    {
        ostringstream sol_name, fomsol_name;
        sol_name << "nldiff-final" << id_param << "." << setfill('0') << setw(6) << myid;
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        p_gf.Save(osol);

        fomsol_name << "nldiff-fom-values-final" << id_param << "." << setfill('0') << setw(6) << myid;
        ofstream fomsol(fomsol_name.str().c_str());
        fomsol.precision(precision);
        for (int i = 0; i < N2; ++i)
        {
            fomsol << (*p_W)[i] << std::endl;
        }
    }

    // 12. Free the used memory.
    delete pmesh;
    delete romop;

    delete p_W;

    totalTimer.Stop();
    if (myid == 0) cout << "Elapsed time for entire simulation " << totalTimer.RealTime() << endl;

    MPI_Finalize();
    return 0;
}

NonlinearDiffusionOperator::NonlinearDiffusionOperator(ParFiniteElementSpace &fR, ParFiniteElementSpace &fW,
        const double rel_tol, const double abs_tol,
        const int iter, const Vector &p)
    : TimeDependentOperator(fR.GetTrueVSize() + fW.GetTrueVSize(), 0.0),
      fespace_R(fR), fespace_W(fW), M(NULL), C(NULL), Bmat(NULL), BTmat(NULL), Mprime(NULL), current_dt(0.0),
      newton_solver(fW.GetComm()), M_solver(NULL), C_solver(fW.GetComm()), zW(fW.GetTrueVSize()), yR(fR.GetTrueVSize()),
      zR(fR.GetTrueVSize()), p0(height), dpdt_prev(height),
      fullOp(NULL), fullGradient(NULL), fullPrec(NULL)
{
    gradient = new NonlinearDiffusionGradientOperator(fR.GetTrueVSize(), fW.GetTrueVSize());

    linear_solver_rel_tol = 1.0e-14;

    C = new ParBilinearForm(&fespace_W);
    C->AddDomainIntegrator(new MassIntegrator());
    C->Assemble();
    C->Finalize();
    Cmat = C->ParallelAssemble();

    C_solver.iterative_mode = false;
    C_solver.SetRelTol(linear_solver_rel_tol);
    C_solver.SetAbsTol(0.0);
    C_solver.SetMaxIter(1000);
    C_solver.SetPrintLevel(0);
    C_prec.SetType(HypreSmoother::Jacobi);
    C_solver.SetPreconditioner(C_prec);
    C_solver.SetOperator(*Cmat);

    ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(&fespace_R, &fespace_W));
    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();
    Bmat = bVarf->ParallelAssemble();
    (*Bmat) *= -1.0;

    BTmat = Bmat->Transpose();

    delete bVarf;

    // See ex5p.cpp for an explanation of block_trueOffsets
    block_trueOffsets.SetSize(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = fespace_R.GetTrueVSize();
    block_trueOffsets[2] = fespace_W.GetTrueVSize();
    block_trueOffsets.PartialSum();

    // SetParameters(p);

    M_prec.SetType(HypreSmoother::Jacobi);

    // Set the newton solve parameters

    //Solver *J_prec = new DSmoother(1);
    J_gmres = new GMRESSolver(MPI_COMM_WORLD);
    J_gmres->SetRelTol(linear_solver_rel_tol);
    J_gmres->SetAbsTol(0.0);
    J_gmres->SetMaxIter(1000);
    J_gmres->SetPrintLevel(2);
    // TODO: precondition, for an efficient FOM solver.
    // For example, see ex5p and miniapps/solvers/block-solvers.cpp in MFEM.
    // J_gmres->SetPreconditioner(*J_prec);

    newton_solver.iterative_mode = true;
    newton_solver.SetSolver(*J_gmres);
    newton_solver.SetOperator(*this);
    newton_solver.SetPrintLevel(1);
    newton_solver.SetRelTol(rel_tol);
    newton_solver.SetAbsTol(abs_tol);
    newton_solver.SetMaxIter(iter);

    dpdt_prev = 0.0;
}

void NonlinearDiffusionOperator::SetBTV(const CAROM::Matrix *V, CAROM::Matrix *BTV) const
{
    const int ncol = BTV->numColumns();
    const int nw = zW.Size();
    const int nr = zR.Size();

    MFEM_VERIFY(V->numRows() == nw && BTV->numRows() == nr && V->numColumns() >= ncol, "");

    for (int k=0; k<ncol; ++k)
    {
        for (int i=0; i<nw; ++i)
            zW[i] = (*V)(i,k);

        BTmat->Mult(zW, zR);

        for (int i=0; i<nr; ++i)
            (*BTV)(i,k) = zR[i];
    }
}

void NonlinearDiffusionOperator::Mult(const Vector &dp_dt, Vector &res) const
{
    Mult_FullSystem(dp_dt, res);
}

void NonlinearDiffusionOperator::GetSource(Vector& s) const
{
    // Set grid function for f
    ParGridFunction f_gf(&fespace_W);

    FunctionCoefficient f(SourceFunction);
    f.SetTime(GetTime());
    f_gf.ProjectCoefficient(f);
    f_gf.GetTrueDofs(s);
}

void NonlinearDiffusionOperator::Mult_FullSystem(const Vector &dp_dt, Vector &res) const
{
    // Compute:
    //    [   Mv + B^T p   ], with p = p0 + dt*dp_dt
    //    [C dp_dt - Bv - f]       v = v0 + dt*dv_dt

    GetSource(zW);

    Vector p(p0);
    p.Add(current_dt, dp_dt);

    SetParameters(p);  // Create fullOp

    fullOp->Mult(p, res);  // Sets the first block row of res. The second block row is computed below.

    Vector p_R(p.GetData() + block_trueOffsets[0], block_trueOffsets[1]-block_trueOffsets[0]);
    Vector pt_W(dp_dt.GetData() + block_trueOffsets[1], block_trueOffsets[2]-block_trueOffsets[1]);
    Vector res_W(res.GetData() + block_trueOffsets[1], block_trueOffsets[2]-block_trueOffsets[1]);

    res_W = pt_W;
    res_W.Add(-1.0, zW);  // -= f
    Cmat->Mult(res_W, zW);  // = C dp_dt - Cf

    res_W = zW;

    Bmat->Mult(p_R, zW);  // Bv
    res_W.Add(-1.0, zW);  // -= Bv
}

void NonlinearDiffusionOperator::ImplicitSolve(const double dt,
        const Vector &p, Vector &dp_dt)
{
    // Solve the equation:
    //    dp_dt = C^{-1} (f - B M(p + dt dp_dt)^{-1} B^T (p + dt dp_dt)), in the Schur complement case
    // for dp_dt

    current_dt = dt;
    // MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

    p0 = p;

    // Set the initial guess for dp_dt, to be used by newton_solver.
    //dp_dt = 0.0;
    dp_dt = dpdt_prev;

    Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
    newton_solver.Mult(zero, dp_dt);

    // MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
    if (newton_solver.GetConverged())
        dpdt_prev = dp_dt;
    else
    {
        dp_dt = 0.0;  // Zero update in SDIRK Step() function.
        newtonFailure = true;
    }
}

Operator &NonlinearDiffusionOperator::GetGradient(const Vector &p) const
{
    // Note that if a matrix A depends on a parameter t, then dA^{-1}/dt = -A^{-1} dA/dt A^{-1}.
    // (d/dp) M(p)^{-1} = -M(p)^{-1} M(a'(p)) M(p)^{-1}

    // Gradient is C^{-1} B M(a(p))^{-1} B^T - C^{-1} B M(p)^{-1} M(a'(p)) M(p)^{-1} B^T, Schur complement case.

    return *fullGradient;
}

void NonlinearDiffusionOperator::SetParameters(const Vector &p) const
{
    // Set grid function for a(p)
    ParGridFunction p_gf(&fespace_W);

    {
        Vector p_W(p.GetData() + block_trueOffsets[1], block_trueOffsets[2]-block_trueOffsets[1]);
        p_gf.SetFromTrueDofs(p_W);
    }

    GridFunctionCoefficient p_coeff(&p_gf);
    TransformedCoefficient a_coeff(&p_coeff, NonlinearCoefficient);
    TransformedCoefficient aprime_coeff(&p_coeff, NonlinearCoefficientDerivative);
    SumCoefficient a_plus_aprime_coeff(a_coeff, aprime_coeff);

    delete M;
    // TODO: delete Mmat?
    M = new ParBilinearForm(&fespace_R);

    M->AddDomainIntegrator(new VectorFEMassIntegrator(a_coeff));
    M->Assemble();
    M->Finalize();
    Mmat = M->ParallelAssemble();

    delete Mprime;
    Mprime = new ParBilinearForm(&fespace_R);

    Mprime->AddDomainIntegrator(new VectorFEMassIntegrator(a_plus_aprime_coeff));

    Mprime->Assemble();
    Mprime->FormSystemMatrix(ess_Rdof_list, Mprimemat);

    Mprimemat *= current_dt;

    delete M_solver;
    M_solver = new CGSolver(fespace_R.GetComm());

    M_solver->iterative_mode = false;
    M_solver->SetRelTol(linear_solver_rel_tol);
    M_solver->SetAbsTol(0.0);
    M_solver->SetMaxIter(1000);
    M_solver->SetPrintLevel(-1);
    M_solver->SetPreconditioner(M_prec);

    {
        M_solver->SetOperator(Mprimemat);

        delete fullOp;
        fullOp = new BlockOperator(block_trueOffsets);
        fullOp->SetBlock(0, 0, Mmat);
        fullOp->SetBlock(0, 1, BTmat);
        fullOp->SetBlock(1, 0, Bmat, -1.0);
        fullOp->SetBlock(1, 1, Cmat);

        delete fullGradient;
        fullGradient = new BlockOperator(block_trueOffsets);
        fullGradient->SetBlock(0, 0, &Mprimemat);
        fullGradient->SetBlock(0, 1, BTmat, current_dt);
        fullGradient->SetBlock(1, 0, Bmat, -current_dt);
        fullGradient->SetBlock(1, 1, Cmat);

        delete fullPrec;
        fullPrec = new BlockDiagonalPreconditioner(block_trueOffsets);
        fullPrec->SetDiagonalBlock(0, M_solver);
        fullPrec->SetDiagonalBlock(1, &C_solver);
        J_gmres->SetPreconditioner(*fullPrec);
    }

    // EQP

    /*
    // Testing FOM mass matrix assembly
    SparseMatrix *amat = Assemble_VectorFEMassIntegrator(&a_coeff, &fespace_R);
    SparseMatrix Mdiag;
    Mmat->GetDiag(Mdiag);

    const double nrm1 = Mdiag.MaxNorm();
    const double nrm2 = amat->MaxNorm();

    SparseMatrix *diff = Add(1.0, Mdiag, -1.0, *amat);

    const double nrm3 = diff->MaxNorm();

    cout << "M matrix max norms: " << nrm1 << ", " << nrm2
	 << ", diff norm " << nrm3 << endl;

    delete amat;
    delete diff;
    */

    // Assemble matrix used in NNLS
}

NonlinearDiffusionOperator::~NonlinearDiffusionOperator()
{
    delete M;
    delete Mprime;
    delete C;
    delete Bmat;
    delete J_gmres;
}

void Compute_CtAB(const HypreParMatrix* A,
                  const CAROM::Matrix& B,  // Distributed matrix.
                  const CAROM::Matrix& C,  // Distributed matrix.
                  CAROM::Matrix* CtAB)     // Non-distributed (local) matrix, computed identically and redundantly on every process.
{
    MFEM_VERIFY(B.distributed() && C.distributed() && !CtAB->distributed(), "");

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int num_rows = B.numRows();
    const int num_cols = B.numColumns();
    const int num_rows_A = A->GetNumRows();

    MFEM_VERIFY(C.numRows() == num_rows_A, "");

    Vector Bvec(num_rows);
    Vector ABvec(num_rows_A);

    CAROM::Matrix AB(num_rows_A, num_cols, true);

    for (int i = 0; i < num_cols; ++i) {
        for (int j = 0; j < num_rows; ++j) {
            Bvec[j] = B(j, i);
        }
        A->Mult(Bvec, ABvec);
        for (int j = 0; j < num_rows_A; ++j) {
            AB(j, i) = ABvec[j];
        }
    }

    C.transposeMult(AB, CtAB);
}

RomOperator::RomOperator(NonlinearDiffusionOperator *fom_, NonlinearDiffusionOperator *fomSp_, const int rrdim_, const int rwdim_,
                         const int nldim_, CAROM::SampleMeshManager *smm_,
                         const CAROM::Matrix* V_R_, const CAROM::Matrix* U_R_, const CAROM::Matrix* V_W_,
                         const CAROM::Matrix *Bsinv,
                         const double newton_rel_tol, const double newton_abs_tol, const int newton_iter,
                         const CAROM::Matrix* S_, const CAROM::Matrix *Ssinv_,
                         const int myid, const bool hyperreduce_source_)
    : TimeDependentOperator(rrdim_ + rwdim_, 0.0),
      newton_solver(),
      fom(fom_), fomSp(fomSp_), BR(NULL), rrdim(rrdim_), rwdim(rwdim_), nldim(nldim_),
      smm(smm_),
      nsamp_R(smm_->GetNumVarSamples("V")), nsamp_S(hyperreduce_source_ ? smm_->GetNumVarSamples("S") : 0),
      V_R(*V_R_), U_R(U_R_), V_W(*V_W_), VTU_R(rrdim_, nldim_, false),
      y0(height), dydt_prev(height), zY(nldim, false), zN(std::max(nsamp_R, 1), false),
      Vsinv(Bsinv), J(height),
      zS(std::max(nsamp_S, 1), false), zT(std::max(nsamp_S, 1), false), Ssinv(Ssinv_),
      VTCS_W(rwdim, std::max(nsamp_S, 1), false), S(S_),
      VtzR(rrdim_, false), hyperreduce_source(hyperreduce_source_)
{
    dydt_prev = 0.0;

    if (myid == 0)
    {
        zR.SetSize(fomSp_->zR.Size());
        BRsp = new CAROM::Matrix(fomSp->zR.Size(), rrdim, false);
        BWsp = new CAROM::Matrix(fomSp->zW.Size(), rwdim, false);
    }

    V_R.transposeMult(*U_R, VTU_R);

    smm->GatherDistributedMatrixRows("V", V_R, rrdim, *BRsp);
    smm->GatherDistributedMatrixRows("P", V_W, rwdim, *BWsp);

    // Compute BR = V_W^t B V_R and CR = V_W^t C V_W, and store them throughout the simulation.

    BR = new CAROM::Matrix(rwdim, rrdim, false);
    CR = new CAROM::Matrix(rwdim, rwdim, false);
    Compute_CtAB(fom->Bmat, V_R, V_W, BR);
    Compute_CtAB(fom->Cmat, V_W, V_W, CR);

    // The ROM residual is
    // [ V_{R,s}^{-1} M(a(Pst V_W p)) Pst V_R v + V_R^t B^T V_W p ]
    // [ V_W^t C V_W dp_dt - V_W^t B V_R v - V_W^t f ]
    // or, with [v, p] = [V_R yR, V_W yW],
    // [ V_{R,s}^{-1} M(a(Pst V_W yW)) Pst V_R yR + BR^T yW ]
    // [ CR dyW_dt - BR yR - V_W^t f ]
    // The Jacobian with respect to [dyR_dt, dyW_dt], with [yR, yW] = [yR0, yW0] + dt * [dyR_dt, dyW_dt], is
    // [ dt V_{R,s}^{-1} M(a'(Pst V_W yW)) Pst V_R  dt BR^T ]
    // [                 -dt BR                        CR   ]

    if (myid == 0)
    {
        const double linear_solver_rel_tol = 1.0e-14;

        J_gmres = new GMRESSolver;
        J_gmres->SetRelTol(linear_solver_rel_tol);
        J_gmres->SetAbsTol(0.0);
        J_gmres->SetMaxIter(1000);
        J_gmres->SetPrintLevel(1);

        newton_solver.iterative_mode = true;
        newton_solver.SetSolver(*J_gmres);
        newton_solver.SetOperator(*this);
        newton_solver.SetPrintLevel(1);
        newton_solver.SetRelTol(newton_rel_tol);
        newton_solver.SetAbsTol(newton_abs_tol);
        newton_solver.SetMaxIter(newton_iter);

        const int spdim = fomSp->Height();

        psp_librom = new CAROM::Vector(spdim, false);
        psp = new Vector(&((*psp_librom)(0)), spdim);

        // Define sub-vectors of psp.
        psp_R = new Vector(psp->GetData(), fomSp->zR.Size());
        psp_W = new Vector(psp->GetData() + fomSp->zR.Size(), fomSp->zW.Size());

        psp_R_librom = new CAROM::Vector(psp_R->GetData(), psp_R->Size(), false, false);
        psp_W_librom = new CAROM::Vector(psp_W->GetData(), psp_W->Size(), false, false);
    }

    hyperreduce = true;
    sourceFOM = false;

    if (!hyperreduce || sourceFOM)
    {
        const int fdim = fom->Height();

        pfom_librom = new CAROM::Vector(fdim, false);
        pfom = new Vector(&((*pfom_librom)(0)), fdim);

        // Define sub-vectors of pfom.
        pfom_R = new Vector(pfom->GetData(), fom->zR.Size());
        pfom_W = new Vector(pfom->GetData() + fom->zR.Size(), fom->zW.Size());

        pfom_R_librom = new CAROM::Vector(pfom_R->GetData(), pfom_R->Size(), false, false);
        pfom_W_librom = new CAROM::Vector(pfom_W->GetData(), pfom_W->Size(), false, false);

        zfomR.SetSize(fom->zR.Size());
        zfomR_librom = new CAROM::Vector(zfomR.GetData(), zfomR.Size(), false, false);

        zfomW.SetSize(fom->zW.Size());
    }

    if (hyperreduce_source)
        Compute_CtAB(fom->Cmat, *S, V_W, &VTCS_W);
}

RomOperator::~RomOperator()
{
    delete BR;
    delete CR;
}

void RomOperator::Mult_Hyperreduced(const Vector &dy_dt, Vector &res) const
{
    MFEM_VERIFY(dy_dt.Size() == rrdim + rwdim && res.Size() == rrdim + rwdim, "");

    Vector y(y0);
    y.Add(current_dt, dy_dt);

    // Evaluate the ROM residual:
    // [ V_R^T U_R U_{R,s}^{-1} M(a(Pst V_W yW)) Pst V_R yR + BR^T yW ]
    // [ CR dyW_dt - BR yR - V_W^t C f ]

    CAROM::Vector y_librom(y.GetData(), y.Size(), false, false);
    CAROM::Vector yR_librom(y.GetData(), rrdim, false, false);
    CAROM::Vector yW_librom(y.GetData() + rrdim, rwdim, false, false);

    CAROM::Vector resR_librom(res.GetData(), rrdim, false, false);
    CAROM::Vector resW_librom(res.GetData() + rrdim, rwdim, false, false);

    CAROM::Vector dyW_dt_librom(dy_dt.GetData() + rrdim, rwdim, false, false);

    // 1. Lift p_s+ = B_s+ y
    BRsp->mult(yR_librom, *psp_R_librom);
    BWsp->mult(yW_librom, *psp_W_librom);

    fomSp->SetParameters(*psp);

    fomSp->Mmat->Mult(*psp_R, zR);  // M(a(Pst V_W yW)) Pst V_R yR

    // Select entries out of zR.
    smm->GetSampledValues("V", zR, zN);

    // Note that it would be better to just store VTU_R * Vsinv, but these are small matrices.
#ifdef USE_GNAT
    Vsinv->transposeMult(zN, zY);
#else
    Vsinv->mult(zN, zY);
#endif

    BR->transposeMult(yW_librom, resR_librom);
    VTU_R.multPlus(resR_librom, zY, 1.0);

    // Apply V_W^t C to fsp

    if (sourceFOM)
    {
        fom->GetSource(zfomW);
        zfomW.Neg();

        fom->Cmat->Mult(zfomW, *pfom_W);

        V_W.transposeMult(*pfom_W_librom, resW_librom);

        CR->multPlus(resW_librom, dyW_dt_librom, 1.0);
        BR->multPlus(resW_librom, yR_librom, -1.0);
    }
    else
    {
        CR->mult(dyW_dt_librom, resW_librom);
        BR->multPlus(resW_librom, yR_librom, -1.0);

        fomSp->GetSource(fomSp->zW);

        if (hyperreduce_source)
        {
            // Select entries
            smm->GetSampledValues("S", fomSp->zW, zT);

#ifdef USE_GNAT
            Ssinv->transposeMult(zT, zS);
#else
            Ssinv->mult(zT, zS);
#endif

            // Multiply by the f-basis, followed by C, followed by V_W^T. This is stored in VTCS_W = V_W^T CS.
            VTCS_W.multPlus(resW_librom, zS, -1.0);
        }
        else
        {
            fomSp->Cmat->Mult(fomSp->zW, *psp_W);

            const int nRsp = fomSp->zR.Size();
            const int nWsp = fomSp->zW.Size();
            for (int i=0; i<rwdim; ++i)
                for (int j=0; j<nWsp; ++j)
                    res[rrdim + i] -= (*BWsp)(j, i) * (*psp_W)[j];
        }
    }
}

void RomOperator::Mult_FullOrder(const Vector &dy_dt, Vector &res) const
{
    MFEM_VERIFY(dy_dt.Size() == rrdim + rwdim && res.Size() == rrdim + rwdim, "");

    Vector y(y0);
    y.Add(current_dt, dy_dt);

    // Evaluate the unreduced ROM residual:
    // [ V_R^T M(a(V_W yW)) V_R yR + BR^T yW ]
    // [ CR dyW_dt - BR yR - V_W^t Cf ]

    CAROM::Vector y_librom(y.GetData(), y.Size(), false, false);
    CAROM::Vector yR_librom(y.GetData(), rrdim, false, false);
    CAROM::Vector yW_librom(y.GetData() + rrdim, rwdim, false, false);

    CAROM::Vector resR_librom(res.GetData(), rrdim, false, false);
    CAROM::Vector resW_librom(res.GetData() + rrdim, rwdim, false, false);

    CAROM::Vector dyW_dt_librom(dy_dt.GetData() + rrdim, rwdim, false, false);

    // 1. Lift p_fom = [V_R^T V_W^T]^T y
    V_R.mult(yR_librom, *pfom_R_librom);
    V_W.mult(yW_librom, *pfom_W_librom);

    fom->SetParameters(*pfom);

    fom->Mmat->Mult(*pfom_R, zfomR);  // M(a(V_W yW)) V_R yR
    V_R.transposeMult(*zfomR_librom, VtzR);  // V_R^T M(a(V_W yW)) V_R yR

    BR->transposeMult(yW_librom, resR_librom);
    resR_librom.plusEqAx(1.0, VtzR);

    // Apply V_W^t C to f
    fom->GetSource(zfomW);
    zfomW.Neg();

    fom->Cmat->Mult(zfomW, *pfom_W);

    V_W.transposeMult(*pfom_W_librom, resW_librom);

    CR->multPlus(resW_librom, dyW_dt_librom, 1.0);
    BR->multPlus(resW_librom, yR_librom, -1.0);
}

void RomOperator::Mult(const Vector &dy_dt, Vector &res) const
{
    if (hyperreduce)
        Mult_Hyperreduced(dy_dt, res);
    else
        Mult_FullOrder(dy_dt, res);
}

void RomOperator::ImplicitSolve(const double dt, const Vector &y, Vector &dy_dt)
{
    y0 = y;

    current_dt = dt;
    fomSp->SetTime(GetTime());
    fomSp->current_dt = dt;

    if (!hyperreduce || sourceFOM)
    {
        fom->SetTime(GetTime());
        fom->current_dt = dt;
    }

    // Set the initial guess for dp_dt, to be used by newton_solver.
    //dp_dt = 0.0;
    dy_dt = dydt_prev;

    Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
    newton_solver.Mult(zero, dy_dt);

    // MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
    if (newton_solver.GetConverged())
        dydt_prev = dy_dt;
    else
    {
        dy_dt = 0.0;  // Zero update in SDIRK Step() function.
        //newtonFailure = true;
        MFEM_VERIFY(false, "ROM Newton convergence failure!");
    }
}

// Debugging tool for checking Jacobian
void RomOperator::PrintFDJacobian(const Vector &p) const
{
    const int N = p.Size();
    Vector pp(N);
    Vector r(N);
    Vector r0(N);

    const double d = 1.0e-8;

    DenseMatrix JFD(N);

    Mult(p, r0);

    for (int j=0; j<N; ++j)
    {
        pp = p;
        pp[j] += d;

        Mult(pp, r);

        r -= r0;
        r /= d;

        for (int i=0; i<N; ++i)
            JFD(i,j) = r[i];
    }

    JFD.Print(cout);
}

Operator &RomOperator::GetGradient(const Vector &p) const
{
    // The Jacobian with respect to [dyR_dt, dyW_dt], with [yR, yW] = [yR0, yW0] + dt * [dyR_dt, dyW_dt], is
    // [ dt V_{R,s}^{-1} M(a'(Pst V_W yW)) Pst V_R  dt BR^T ]
    // [                 -dt BR                        CR   ]

    // Compute JR = V_{R,s}^{-1} M(a'(Pst V_W yW)) Pst V_R, assuming M(a'(Pst V_W yW)) is already stored in fomSp->Mprimemat,
    // which is computed in fomSp->SetParameters, which was called by RomOperator::Mult, which was called by newton_solver
    // before this call to GetGradient. Note that V_R restricted to the sample matrix is already stored in Bsp.

    CAROM::Vector r(nldim, false);
    CAROM::Vector c(rrdim, false);
    CAROM::Vector z(nsamp_R, false);

    for (int i=0; i<rrdim; ++i)
    {
        if (hyperreduce)
        {
            // Compute the i-th column of M(a'(Pst V_W yW)) Pst V_R.
            for (int j=0; j<psp_R->Size(); ++j)
                (*psp_R)[j] = (*BRsp)(j,i);

            fomSp->Mprimemat.Mult(*psp_R, zR);

            smm->GetSampledValues("V", zR, z);

            // Note that it would be better to just store VTU_R * Vsinv, but these are small matrices.

#ifdef USE_GNAT
            Vsinv->transposeMult(z, r);
#else
            Vsinv->mult(z, r);
#endif

            VTU_R.mult(r, c);
        }
        else
        {
            // Compute the i-th column of V_R^T M(a'(V_W yW)) V_R.
            for (int j=0; j<pfom_R->Size(); ++j)
                (*pfom_R)[j] = V_R(j,i);

            fom->Mprimemat.Mult(*pfom_R, zfomR);
            V_R.transposeMult(*zfomR_librom, c);  // V_R^T M(a'(V_W yW)) V_R(:,i)
        }

        for (int j=0; j<rrdim; ++j)
            J(j, i) = c(j);  // This already includes a factor of current_dt, from Mprimemat.

        for (int j=0; j<rwdim; ++j)
        {
            J(rrdim + j, i) = -current_dt * (*BR)(j,i);
            J(i, rrdim + j) = current_dt * (*BR)(j,i);
        }
    }

    for (int i=0; i<rwdim; ++i)
    {
        for (int j=0; j<rwdim; ++j)
        {
            J(rrdim + j, rrdim + i) = (*CR)(j,i);
        }
    }

    // TODO: define Jacobian block-wise rather than entry-wise?
    /*
    gradient->SetBlock(0, 0, JR, current_dt);
    gradient->SetBlock(0, 1, BRT, current_dt);
    gradient->SetBlock(1, 0, BR, -current_dt);
    gradient->SetBlock(1, 1, CR);
    */

    // PrintFDJacobian(p);

    return J;
}

double InitialTemperature(const Vector &x)
{
    if (problem == INIT_STEP)
    {
        if (0.5 - step_half < x[0] && x[0] < 0.5 + step_half && 0.5 - step_half < x[1] && x[1] < 0.5 + step_half)
            return 1.0;
        else
            return 0.0;
    }
    else
        return 0.0;
}

double ExactSolution(const Vector &x, const double t)
{
    const double pi = acos(-1.0);
    const double pi2 = 2.0 * acos(-1.0);
    return sin(4.0 * pi * t) * sin(pi2 * x[0]) * sin(pi2 * x[1]);
}

double SourceFunction_linear(const Vector &x, const double t)
{
    // dp/dt + div(v) = f, grad p = -a(p) v
    // p(x,y) = sin(2 pi x) sin(2 pi y) sin(4 pi t)
    // a(p) = 1
    // Set cx = cos(2 pi x), sy = sin(2 pi y), etc. Then v = -2 pi [cx sy st, sx cy st]
    // div(v) = 8 pi^2 sx sy st

    // dp/dt + div(v) = 4 pi sin(2 pi x) sin(2 pi y) cos(4 pi t) + 8 pi^2 sx sy st

    //return 0.0;

    const double pi = acos(-1.0);
    const double pi2 = 2.0 * pi;

    const double sx = sin(pi2 * x[0]);
    const double sy = sin(pi2 * x[1]);

    const double st = sin(4.0 * pi * t);
    const double ct = cos(4.0 * pi * t);

    return (4.0 * pi * ct * sx * sy) + (st * 8.0 * pi * pi * sx * sy);
}

// a(p) = c + p, where c = diffusion_c
double SourceFunction_cpu(const Vector &x, const double t)
{
    // dp/dt + div(v) = f, grad p = -a(p) v
    // p(x,y) = sin(2 pi x) sin(2 pi y) sin(4 pi t)
    // a(p) = c + sin(2 pi x) sin(2 pi y) sin(4 pi t)
    // Set cx = cos(2 pi x), sy = sin(2 pi y), etc. Then v = -2 pi st/(2+sx sy st) * [cx sy, sx cy] = -2 pi st/a * [cx sy, sx cy]

    // div(v) = 4 pi^2 cx^2 sy^2 st^2 / a^2 + 4 pi^2 st / a * sx sy         (dv_x/dx)
    //            + 4 pi^2 sx^2 cy^2 st^2 / a^2 + 4 pi^2 st / a * sx sy     (dv_y/dy)
    //        = 4 pi^2 st / a * [(cx sy)^2 st / a + 2 sx sy + (sx cy)^2 st / a]

    // dp/dt + div(v) = 4 pi sx sy ct + 4 pi^2 st / a * [(cx sy)^2 st / a + 2 sx sy + (sx cy)^2 st / a]

    //return 0.0;

    const double pi = acos(-1.0);
    const double pi2 = 2.0 * acos(-1.0);

    const double cx = cos(pi2 * x[0]);
    const double cy = cos(pi2 * x[1]);

    const double sx = sin(pi2 * x[0]);
    const double sy = sin(pi2 * x[1]);

    const double st = sin(4.0 * pi * t);
    const double ct = cos(4.0 * pi * t);

    const double a = diffusion_c + (sx * sy * st);

    return (4.0 * pi * ct * sx * sy) + (st * 4.0 * pi * pi * ((cx*sy*cx*sy*st/a) + (2.0*sx*sy) + (sx*cy*sx*cy*st/a)) / a);
}

double SourceFunction(const Vector &x, const double t)
{
    if (nonlinear_problem)
    {
        if (problem == INIT_STEP)
            return 0.0;
        else
            return SourceFunction_cpu(x, t);
    }
    else
        return SourceFunction_linear(x, t);
}

double NonlinearCoefficient(const double p)
{
    if (nonlinear_problem)
        return diffusion_c + p;
    else
        return 1.0;
}

double NonlinearCoefficientDerivative(const double p)
{
    if (nonlinear_problem)
        return 1.0;
    else
        return 0.0;
}
