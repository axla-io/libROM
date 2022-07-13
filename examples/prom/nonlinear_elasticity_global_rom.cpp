
#include "mfem.hpp"
#include "linalg/Vector.h"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "hyperreduction/DEIM.h"
#include "hyperreduction/GNAT.h"
#include "hyperreduction/S_OPT.h"
#include "mfem/SampleMesh.hpp"

#include <memory>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class ReducedSystemOperator;



class HyperelasticOperator : public TimeDependentOperator
{
protected:
    ParFiniteElementSpace& fespace;
    Array<int> ess_tdof_list;

    ParBilinearForm M, S;
    ParNonlinearForm H;
    double viscosity;
    HyperelasticModel* model;

    HypreParMatrix* Mmat; // Mass matrix from ParallelAssemble()
    CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
    HypreSmoother M_prec; // Preconditioner for the mass matrix M

    /** Nonlinear operator defining the reduced backward Euler equation for the
        velocity. Used in the implementation of method ImplicitSolve. */
    ReducedSystemOperator* reduced_oper;

    /// Newton solver for the reduced backward Euler equation
    NewtonSolver newton_solver;

    /// Solver for the Jacobian solve in the Newton method
    Solver* J_solver;
    /// Preconditioner for the Jacobian solve in the Newton method
    Solver* J_prec;

    mutable Vector z; // auxiliary vector

    Vector dvxdt_prev; // for computing sample time

public:
    HyperelasticOperator(ParFiniteElementSpace& f, Array<int>& ess_bdr,
        double visc, double mu, double K);

    /// Compute the right-hand side of the ODE system.
    virtual void Mult(const Vector& vx, Vector& dvx_dt) const;
    /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector& x, Vector& k);

    double ElasticEnergy(const ParGridFunction& x) const;
    double KineticEnergy(const ParGridFunction& v) const;
    void GetElasticEnergyDensity(const ParGridFunction& x,
        ParGridFunction& w) const;

    void CopyDvxDt(Vector& dvxdt) const
    {
        dvxdt = dvxdt_prev;
    }


    virtual ~HyperelasticOperator();
};


/** Nonlinear operator of the form:
    k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
    where M and S are given BilinearForms, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class ReducedSystemOperator : public Operator
{
private:
    ParBilinearForm* M, * S;
    ParNonlinearForm* H;
    mutable HypreParMatrix* Jacobian;
    double dt;
    const Vector* v, * x;
    mutable Vector w, z;
    const Array<int>& ess_tdof_list;

public:
    ReducedSystemOperator(ParBilinearForm* M_, ParBilinearForm* S_,
        ParNonlinearForm* H_, const Array<int>& ess_tdof_list);

    /// Set current dt, v, x values - needed to compute action and Jacobian.
    void SetParameters(double dt_, const Vector* v_, const Vector* x_);

    /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
    virtual void Mult(const Vector& k, Vector& y) const;

    /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
    virtual Operator& GetGradient(const Vector& k) const;

    virtual ~ReducedSystemOperator();
};


// TODO: Change to be Hyperelastic operator
class RomOperator : public TimeDependentOperator
{
private:
    int rrdim, rwdim, nldim;
    int nsamp_R, nsamp_S;
    double current_dt;
    bool oversampling;
    NewtonSolver newton_solver;
    GMRESSolver* J_gmres;
    CAROM::Matrix* BRsp, * BWsp;
    CAROM::Vector* psp_librom, * psp_R_librom, * psp_W_librom;
    Vector* psp;
    Vector* psp_R;
    Vector* psp_W;
    mutable Vector zR;
    mutable CAROM::Vector zY;
    mutable CAROM::Vector zN;
    const CAROM::Matrix* Vsinv;

    // Data for source function
    const CAROM::Matrix* Ssinv;
    mutable CAROM::Vector zS;
    mutable CAROM::Vector zT;
    const CAROM::Matrix* S;

    mutable DenseMatrix J;

    bool hyperreduce, hyperreduce_source;
    bool sourceFOM;

    CAROM::Vector* pfom_librom, * pfom_R_librom, * pfom_W_librom;
    Vector* pfom;
    Vector* pfom_R;
    Vector* pfom_W;
    mutable Vector zfomR;
    mutable Vector zfomW;
    CAROM::Vector* zfomR_librom;
    mutable CAROM::Vector VtzR;

    CAROM::SampleMeshManager* smm;

    void PrintFDJacobian(const Vector& p) const;

protected:
    CAROM::Matrix* BR;
    CAROM::Matrix* CR;
    const CAROM::Matrix* U_R;
    Vector y0;
    Vector dydt_prev;
    NonlinearDiffusionOperator* fom;
    NonlinearDiffusionOperator* fomSp;

public:
    RomOperator(NonlinearDiffusionOperator* fom_,
        NonlinearDiffusionOperator* fomSp_,
        const int rrdim_, const int rwdim_, const int nldim_,
        CAROM::SampleMeshManager* smm_,
        const CAROM::Matrix* V_R_, const CAROM::Matrix* U_R_, const CAROM::Matrix* V_W_,
        const CAROM::Matrix* Bsinv,
        const double newton_rel_tol, const double newton_abs_tol, const int newton_iter,
        const CAROM::Matrix* S_, const CAROM::Matrix* Ssinv_,
        const int myid, const bool hyperreduce_source, const bool oversampling);

    virtual void Mult(const Vector& y, Vector& dy_dt) const;
    void Mult_Hyperreduced(const Vector& y, Vector& dy_dt) const;
    void Mult_FullOrder(const Vector& y, Vector& dy_dt) const;

    /** Solve the Backward-Euler equation: k = f(p + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector& y, Vector& dy_dt);

    virtual Operator& GetGradient(const Vector& y) const;

    CAROM::Matrix V_W, V_R, VTU_R;

    CAROM::Matrix VTCS_W;

    virtual ~RomOperator();
};





/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
    HyperelasticModel& model;
    const ParGridFunction& x;
    DenseMatrix            J;

public:
    ElasticEnergyCoefficient(HyperelasticModel& m, const ParGridFunction& x_)
        : model(m), x(x_) { }
    virtual double Eval(ElementTransformation& T, const IntegrationPoint& ip);
    virtual ~ElasticEnergyCoefficient() { }
};

void InitialDeformation(const Vector& x, Vector& y);

void InitialVelocity(const Vector& x, Vector& v);

void visualize(ostream& out, ParMesh* mesh, ParGridFunction* deformed_nodes,
    ParGridFunction* field, const char* field_name = NULL,
    bool init_vis = false);



// TODO: move this to the library?
CAROM::Matrix* GetFirstColumns(const int N, const CAROM::Matrix* A)
{
    CAROM::Matrix* S = new CAROM::Matrix(A->numRows(), std::min(N, A->numColumns()),
        A->distributed());
    for (int i = 0; i < S->numRows(); ++i)
    {
        for (int j = 0; j < S->numColumns(); ++j)
            (*S)(i, j) = (*A)(i, j);
    }

    // delete A;  // TODO: find a good solution for this.
    return S;
}

// TODO: move this to the library?
void BasisGeneratorFinalSummary(CAROM::BasisGenerator* bg,
    const double energyFraction, int& cutoff, const std::string cutoffOutputPath)
{
    const int rom_dim = bg->getSpatialBasis()->numColumns();
    const CAROM::Vector* sing_vals = bg->getSingularValues();

    MFEM_VERIFY(rom_dim <= sing_vals->dim(), "");

    double sum = 0.0;
    for (int sv = 0; sv < sing_vals->dim(); ++sv) {
        sum += (*sing_vals)(sv);
    }

    vector<double> energy_fractions = { 0.9999, 0.999, 0.99, 0.9 };
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
                    << sv + 1 << " of " << sing_vals->dim() << " basis vectors" << endl;
                energy_fractions.pop_back();
            }
            else
            {
                break;
            }
        }

        if (!reached_cutoff && partialSum / sum > energyFraction)
        {
            cutoff = sv + 1;
            reached_cutoff = true;
        }
    }

    if (!reached_cutoff) cutoff = sing_vals->dim();
    outfile << "Take first " << cutoff << " of " << sing_vals->dim() <<
        " basis vectors" << endl;
    outfile.close();
}

void MergeBasis(const int dimFOM, const int nparam, const int max_num_snapshots,
    std::string name)
{
    MFEM_VERIFY(nparam > 0, "Must specify a positive number of parameter sets");

    bool update_right_SV = false;
    bool isIncremental = false;

    CAROM::Options options(dimFOM, nparam * max_num_snapshots, 1, update_right_SV);
    CAROM::BasisGenerator generator(options, isIncremental, "basis" + name);

    for (int paramID = 0; paramID < nparam; ++paramID)
    {
        std::string snapshot_filename = "basis" + std::to_string(
            paramID) + "_" + name + "_snapshot";
        generator.loadSamples(snapshot_filename, "snapshot");
    }

    generator.endSamples(); // save the merged basis file

    int cutoff = 0;
    BasisGeneratorFinalSummary(&generator, 0.9999, cutoff, "mergedSV_" + name);
}

int main(int argc, char* argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);



    // 2. Parse command-line options.
    const char* mesh_file = "../data/beam-quad.mesh";
    int ser_ref_levels = 2;
    int par_ref_levels = 0;
    int order = 2;
    int ode_solver_type = 3;
    double t_final = 300.0;
    double dt = 3.0;
    double visc = 1e-2;
    double mu = 0.25;
    double K = 5.0;
    bool adaptive_lin_rtol = true;
    bool visualization = true;
    bool visit = false;
    int vis_steps = 1;

    // ROM parameters
    bool offline = true; // debug mode
    bool merge = false;
    bool online = false;
    bool use_sopt = false;
    int num_samples_req = -1;

    int nsets = 0;

    int id_param = 0;


    // Number of basis vectors to use
    int rdim = -1;
    int nldim = -1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
        "Mesh file to use.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
        "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
        "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
        "Order (degree) of the finite elements.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
        "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
        "            11 - Forward Euler, 12 - RK2,\n\t"
        "            13 - RK3 SSP, 14 - RK4."
        "            22 - Implicit Midpoint Method,\n\t"
        "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
    args.AddOption(&t_final, "-tf", "--t-final",
        "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
        "Time step.");
    args.AddOption(&visc, "-v", "--viscosity",
        "Viscosity coefficient.");
    args.AddOption(&mu, "-mu", "--shear-modulus",
        "Shear modulus in the Neo-Hookean hyperelastic model.");
    args.AddOption(&K, "-K", "--bulk-modulus",
        "Bulk modulus in the Neo-Hookean hyperelastic model.");
    args.AddOption(&adaptive_lin_rtol, "-alrtol", "--adaptive-lin-rtol",
        "-no-alrtol", "--no-adaptive-lin-rtol",
        "Enable or disable adaptive linear solver rtol.");
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
    args.AddOption(&use_sopt, "-sopt", "--sopt", "-no-sopt", "--no-sopt",
        "Use S-OPT sampling instead of DEIM for the hyperreduction.");
    args.AddOption(&num_samples_req, "-nsr", "--nsr",
        "number of samples we want to select for the sampling algorithm.");

    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    const bool check = (offline && !merge && !online) || (!offline && merge
        && !online) || (!offline && !merge && online);
    MFEM_VERIFY(check, "only one of offline, merge, or online must be true!");

    const bool hyperreduce_source = (problem != INIT_STEP);

    StopWatch solveTimer, totalTimer;
    totalTimer.Start();



    // 3. Read the serial mesh from the given mesh file on all processors. We can
    //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    Mesh* mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Define the ODE solver used for time integration. Several implicit
    //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    //    explicit Runge-Kutta methods are available.
    ODESolver* ode_solver;
    switch (ode_solver_type)
    {
        // Implicit L-stable methods
    case 1:
        ode_solver = new BackwardEulerSolver;
        break;
    case 2:
        ode_solver = new SDIRK23Solver(2);
        break;
    case 3:
        ode_solver = new SDIRK33Solver;
        break;
        // Explicit methods
    case 11:
        ode_solver = new ForwardEulerSolver;
        break;
    case 12:
        ode_solver = new RK2Solver(0.5);
        break; // midpoint method
    case 13:
        ode_solver = new RK3SSPSolver;
        break;
    case 14:
        ode_solver = new RK4Solver;
        break;
    case 15:
        ode_solver = new GeneralizedAlphaSolver(0.5);
        break;
        // Implicit A-stable methods (not L-stable)
    case 22:
        ode_solver = new ImplicitMidpointSolver;
        break;
    case 23:
        ode_solver = new SDIRK23Solver;
        break;
    case 24:
        ode_solver = new SDIRK34Solver;
        break;
    default:
        if (myid == 0)
        {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        }
        delete mesh;
        MPI_Finalize();
        return 3;
    }

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
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }


    // 7. Define the parallel vector finite element spaces representing the mesh
    //    deformation x_gf, the velocity v_gf, and the initial configuration,
    //    x_ref. Define also the elastic energy density, w_gf, which is in a
    //    discontinuous higher-order space. Since x and v are integrated in time
    //    as a system, we group them together in block vector vx, on the unique
    //    parallel degrees of freedom, with offsets given by array true_offset.
    H1_FECollection fe_coll(order, dim);
    ParFiniteElementSpace fespace(pmesh, &fe_coll, dim);

    HYPRE_BigInt glob_size = fespace.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of velocity/deformation unknowns: " << glob_size << endl;
    }
    int true_size = fespace.TrueVSize();
    Array<int> true_offset(3);
    true_offset[0] = 0;
    true_offset[1] = true_size;
    true_offset[2] = 2 * true_size;

    BlockVector vx(true_offset);
    ParGridFunction v_gf, x_gf;
    v_gf.MakeTRef(&fespace, vx, true_offset[0]); // Associate a new FiniteElementSpace and new true-dof data with the GridFunction. 
    x_gf.MakeTRef(&fespace, vx, true_offset[1]);

    ParGridFunction x_ref(&fespace);
    pmesh->GetNodes(x_ref);

    L2_FECollection w_fec(order + 1, dim);
    ParFiniteElementSpace w_fespace(pmesh, &w_fec);
    ParGridFunction w_gf(&w_fespace);


    // Basis params
    bool update_right_SV = false;
    bool isIncremental = false;
    const std::string basisFileName = "basis" + std::to_string(id_param);
    int max_num_snapshots = int(t_final / dt) + 2;

    // The merge phase
    if (merge)
    {
        totalTimer.Clear();
        totalTimer.Start();

        MergeBasis(R_space.GetTrueVSize(), nsets, max_num_snapshots, "VX");

        if (hyperreduce_source)
        {
            MergeBasis(W_space.GetTrueVSize(), nsets, max_num_snapshots, "H");
        }

        totalTimer.Stop();
        if (myid == 0)
        {
            printf("Elapsed time for merging and building ROM basis: %e second\n",
                totalTimer.RealTime());
        }
        MPI_Finalize();
        return 0;
    }


    // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
    //    boundary conditions on a beam-like mesh (see description above).
    VectorFunctionCoefficient velo(dim, InitialVelocity);
    v_gf.ProjectCoefficient(velo);
    v_gf.SetTrueVector();
    VectorFunctionCoefficient deform(dim, InitialDeformation);
    x_gf.ProjectCoefficient(deform);
    x_gf.SetTrueVector();

    v_gf.SetFromTrueVector();
    x_gf.SetFromTrueVector();

    Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed


    Vector dvxdt;

    // 9. Initialize the hyperelastic operator, the GLVis visualization and print
    //    the initial energies.
    HyperelasticOperator oper(fespace, ess_bdr, visc, mu, K);
    HyperelasticOperator* soper = 0;

    socketstream vis_v, vis_w;
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport = 19916;
        vis_v.open(vishost, visport);
        vis_v.precision(8);
        visualize(vis_v, pmesh, &x_gf, &v_gf, "Velocity", true);
        // Make sure all ranks have sent their 'v' solution before initiating
        // another set of GLVis connections (one from each rank):
        MPI_Barrier(pmesh->GetComm());
        vis_w.open(vishost, visport);
        if (vis_w)
        {
            oper.GetElasticEnergyDensity(x_gf, w_gf);
            vis_w.precision(8);
            visualize(vis_w, pmesh, &x_gf, &w_gf, "Elastic energy density", true);
        }
    }



    // Create data collection for solution output: either VisItDataCollection for
    // ascii data files, or SidreDataCollection for binary data files.
    DataCollection* dc = NULL;
    if (visit)
    {
        if (offline)
            dc = new VisItDataCollection("nlelast-fom", pmesh);
        else
            dc = new VisItDataCollection("nlelast-rom", pmesh);

        dc = new VisItDataCollection("Nonlinear_Elasticity", pmesh);
        dc->SetPrecision(8);
        // To save the mesh using MFEM's parallel mesh format:
        // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc->RegisterField("x", &x_gf);
        dc->RegisterField("v", &v_gf);
        dc->SetCycle(0);
        dc->SetTime(0.0);
        dc->Save();
    }

    double ee0 = oper.ElasticEnergy(x_gf);
    double ke0 = oper.KineticEnergy(v_gf);
    if (myid == 0)
    {
        cout << "initial elastic energy (EE) = " << ee0 << endl;
        cout << "initial kinetic energy (KE) = " << ke0 << endl;
        cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;
    }


    // 10. Create pROM object.
    CAROM::BasisGenerator* basis_generator_vx =
        0;  // For the deformation and velocity solution in vector H1

    CAROM::BasisGenerator* basis_generator_H =
        0; // For the non-linear Hyperelasticity operator


    if (offline) {
        CAROM::Options options_vx(fespace.GetTrueVSize(), max_num_snapshots, 1,
            update_right_SV);

        if (hyperreduce_source)
            basis_generator_H = new CAROM::BasisGenerator(options_vx, isIncremental,
                basisFileName + "_H");

        basis_generator_vx = new CAROM::BasisGenerator(options_vx, isIncremental,
            basisFileName + "_VX");

    }


    RomOperator* romop = 0;

    const CAROM::Matrix* vx_librom = 0;
    const CAROM::Matrix* H_librom = 0;

    int nsamp_vx = -1;
    int nsamp_H = -1;

    CAROM::SampleMeshManager* smm = nullptr;


    // The online phase
    if (online)
    {

        // Read bases
        CAROM::BasisReader readerVX("basisVX");
        BR_librom = readerR.getSpatialBasis(0.0);
        if (rrdim == -1)
            rrdim = BR_librom->numColumns();
        else
            BR_librom = GetFirstColumns(rrdim,
                BR_librom);  // TODO: reduce rrdim if too large

        MFEM_VERIFY(BR_librom->numRows() == N1, "");

        if (myid == 0)
            printf("reduced VX dim = %d\n", rrdim);
       

        vector<int> sample_dofs_H;  // Indices of the sampled rows
        vector<int> num_sample_dofs_per_proc_H(num_procs);

        vector<int> sample_dofs_withH;  // Indices of the sampled rows
        CAROM::Matrix* Hsinv = 0;
        vector<int> num_sample_dofs_per_proc_withH;
        CAROM::BasisReader* readerH = 0;
        if (hyperreduce_source)
        {
            readerH = new CAROM::BasisReader("basisH");
            H_librom = readerH->getSpatialBasis(0.0);

            // Compute sample points using DEIM

            if (nsdim == -1)
            {
                nsdim = H_librom->numColumns();
            }

            MFEM_VERIFY(H_librom->numColumns() >= nsdim, "");

            if (H_librom->numColumns() > nsdim)
                H_librom = GetFirstColumns(nsdim, S_librom);

            if (myid == 0)
                printf("reduced H dim = %d\n", nsdim);

            // Now execute the DEIM algorithm to get the sampling information.
            if (num_samples_req != -1)
            {
                nsamp_H = num_samples_req;
            }
            else
            {
                nsamp_H = nsdim;
            }

            Hsinv = new CAROM::Matrix(nsamp_S, nsdim, false);
            sample_dofs_H.resize(nsamp_H);
            if (use_sopt)
            {
                CAROM::S_OPT(H_librom,
                    nsdim,
                    sample_dofs_H,
                    num_sample_dofs_per_proc_H,
                    *Hsinv,
                    myid,
                    num_procs,
                    nsamp_H);
            }
            else if (nsamp_H != nsdim)
            {
                CAROM::GNAT(H_librom,
                    nsdim,
                    sample_dofs_H,
                    num_sample_dofs_per_proc_H,
                    *Hsinv,
                    myid,
                    num_procs,
                    nsamp_H);
            }
            else
            {
                CAROM::DEIM(H_librom,
                    nsdim,
                    sample_dofs_H,
                    num_sample_dofs_per_proc_H,
                    *Hsinv,
                    myid,
                    num_procs);
            }
        }

        // Construct sample mesh

        const int nspaces = 1;
        std::vector<ParFiniteElementSpace*> spfespace(nspaces);
        spfespace[0] = &fespace;

        smm = new CAROM::SampleMeshManager(spfespace);

        vector<int>
            sample_dofs_empty;  // Potential variable in W space has no sample DOFs.
        vector<int> num_sample_dofs_per_proc_empty;
        num_sample_dofs_per_proc_empty.assign(num_procs, 0);

        smm->RegisterSampledVariable("VX", FESPACE, sample_dofs,
            num_sample_dofs_per_proc);

        if (hyperreduce_source)
        {
            
            smm->RegisterSampledVariable("H", FESPACE, sample_dofs_S,
                num_sample_dofs_per_proc_S);
        }

        smm->ConstructSampleMesh();


        // TODO: What does the following code do??
        w = new CAROM::Vector(rrdim + rwdim, false);
        w_W = new CAROM::Vector(rwdim, false);

        // Initialize w = B_W^T p.
        BW_librom->transposeMult(*p_W_librom, *w_W);

        for (int i = 0; i < rrdim; ++i)
            (*w)(i) = 0.0;

        for (int i = 0; i < rwdim; ++i)
            (*w)(rrdim + i) = (*w_W)(i);

        // Note that some of this could be done only on the ROM solver process, but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = new Vector(&((*w)(0)), rrdim + rwdim);



       
        if (myid == 0)
        {
            sp_FEspace = smm->GetSampleFESpace(FESPACE);

            // Initialize sp_p with initial conditions.
            {
                // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
                //    boundary conditions on a beam-like mesh (see description above).

                BlockVector sp_vx(true_offset);
                ParGridFunction sp_v_gf, sp_x_gf;
                sp_v_gf.MakeTRef(&sp_FEspace, sp_vx, true_offset[0]); // Associate a new FiniteElementSpace and new true-dof data with the GridFunction.
                sp_x_gf.MakeTRef(&sp_FEspace, sp_vx, true_offset[1]);


                VectorFunctionCoefficient velo(dim, InitialVelocity);
                sp_v_gf.ProjectCoefficient(velo);
                sp_v_gf.SetTrueVector();
                VectorFunctionCoefficient deform(dim, InitialDeformation);
                sp_x_gf.ProjectCoefficient(deform);
                sp_x_gf.SetTrueVector();

                sp_v_gf.SetFromTrueVector();
                sp_x_gf.SetFromTrueVector();
            }

            // Define operator
            soper = new HyperelasticOperator oper(*sp_FEspace, ess_bdr, visc, mu, K);

        }

        // FIX RomOperator so that it works
        romop = new RomOperator(&oper, soper, rrdim, rwdim, nldim, smm,
            BR_librom, FR_librom, BW_librom,
            Bsinv, newton_rel_tol, newton_abs_tol, newton_iter,
            S_librom, Ssinv, myid, hyperreduce_source, num_samples_req != -1);

        ode_solver.Init(*romop);

        delete readerS;
    }
    else  // fom
        ode_solver.Init(oper);


    // 11. Perform time-integration
    //     (looping over the time iterations, ti, with a time-step dt).
    //     (taking samples and storing it into the pROM object)
    StopWatch fom_timer;

    double t = 0.0;
    vector<double> ts;
    oper.SetTime(t);
    ode_solver->Init(oper);

    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
        fom_timer.Start();

        double dt_real = min(dt, t_final - t);
        ode_solver->Step(vx, t, dt_real);
        last_step = (t >= t_final - 1e-8 * dt);

        fom_timer.Stop();

        if (online)
        {   
            // TODO make this work!
            if (myid == 0)
            {
                ode_solver.Step(*wMFEM, t, dt);
            }

            MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        if (offline)
        {
            // Take sample
            if (basis_generator_vx->isNextSample(t))
            {
                oper.CopyDvxDt(dvxdt);
                basis_generator_vx->takeSample(vx->GetData(), t, dt);
                basis_generator_vx->computeNextSampleTime(vx->GetData(), dvxdt.GetData(), t);

                if (hyperreduce_source)
                {
                    basis_generator_H->takeSample(dvxdt.GetData(), t, dt);

                }

            }
        }
        

        if (last_step || (ti % vis_steps) == 0)
        {
            v_gf.SetFromTrueVector();
            x_gf.SetFromTrueVector();

            double ee = oper.ElasticEnergy(x_gf);
            double ke = oper.KineticEnergy(v_gf);

            if (myid == 0)
            {
                cout << "step " << ti << ", t = " << t << ", EE = " << ee
                    << ", KE = " << ke << ", ΔTE = " << (ee + ke) - (ee0 + ke0) << endl;
            }

            if (visualization)
            {
                visualize(vis_v, pmesh, &x_gf, &v_gf);
                if (vis_w)
                {
                    oper.GetElasticEnergyDensity(x_gf, w_gf);
                    visualize(vis_w, pmesh, &x_gf, &w_gf);
                }
            }

            if (visit)
            {
                dc->SetCycle(ti);
                dc->SetTime(t);
                dc->Save();
            }
        }

    }





    if (offline)
    {
        // Sample final solution, to prevent extrapolation in ROM between the last sample and the end of the simulation.

        oper.CopyDvxDt(dvxdt);
        basis_generator_vx->takeSample(vx->GetData(), t, dt);
        basis_generator_vx->writeSnapshot();

        if (hyperreduce_source)
        {
            basis_generator_H->takeSample(dvxdt.GetData(), t, dt);
            basis_generator_H->writeSnapshot();

        }


        // Terminate the sampling and write out information.
        delete basis_generator_vx;
        delete basis_generator_H;
    }




    // 12. Save the displaced mesh, the velocity and elastic energy.
    {
        v_gf.SetFromTrueVector();
        x_gf.SetFromTrueVector();
        GridFunction* nodes = &x_gf;
        int owns_nodes = 0;
        pmesh->SwapNodes(nodes, owns_nodes);

        ostringstream mesh_name, velo_name, ee_name;
        mesh_name << "deformed." << setfill('0') << setw(6) << myid;
        velo_name << "velocity." << setfill('0') << setw(6) << myid;
        ee_name << "elastic_energy." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);
        pmesh->SwapNodes(nodes, owns_nodes);
        ofstream velo_ofs(velo_name.str().c_str());
        velo_ofs.precision(8);
        v_gf.Save(velo_ofs);
        ofstream ee_ofs(ee_name.str().c_str());
        ee_ofs.precision(8);
        oper.GetElasticEnergyDensity(x_gf, w_gf);
        w_gf.Save(ee_ofs);
    }


    // 16. Free the used memory.
    delete ode_solver;
    delete pmesh;

    MPI_Finalize();





    return 1;
}



void visualize(ostream& out, ParMesh* mesh, ParGridFunction* deformed_nodes,
    ParGridFunction* field, const char* field_name, bool init_vis)
{
    if (!out)
    {
        return;
    }

    GridFunction* nodes = deformed_nodes;
    int owns_nodes = 0;

    mesh->SwapNodes(nodes, owns_nodes);

    out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
    out << "solution\n" << *mesh << *field;

    mesh->SwapNodes(nodes, owns_nodes);

    if (init_vis)
    {
        out << "window_size 800 800\n";
        out << "window_title '" << field_name << "'\n";
        if (mesh->SpaceDimension() == 2)
        {
            out << "view 0 0\n"; // view from top
            out << "keys jl\n";  // turn off perspective and light
        }
        out << "keys cm\n";         // show colorbar and mesh
        out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
    }
    out << flush;
}



ReducedSystemOperator::ReducedSystemOperator(
    ParBilinearForm* M_, ParBilinearForm* S_, ParNonlinearForm* H_,
    const Array<int>& ess_tdof_list_)
    : Operator(M_->ParFESpace()->TrueVSize()), M(M_), S(S_), H(H_),
    Jacobian(NULL), dt(0.0), v(NULL), x(NULL), w(height), z(height),
    ess_tdof_list(ess_tdof_list_)
{ }

void ReducedSystemOperator::SetParameters(double dt_, const Vector* v_,
    const Vector* x_)
{
    dt = dt_;
    v = v_;
    x = x_;
}

void ReducedSystemOperator::Mult(const Vector& k, Vector& y) const
{
    // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
    add(*v, dt, k, w);
    add(*x, dt, w, z);
    H->Mult(z, y);
    M->TrueAddMult(k, y);
    S->TrueAddMult(w, y);
    y.SetSubVector(ess_tdof_list, 0.0);
}

Operator& ReducedSystemOperator::GetGradient(const Vector& k) const
{
    delete Jacobian;
    SparseMatrix* localJ = Add(1.0, M->SpMat(), dt, S->SpMat());
    add(*v, dt, k, w);
    add(*x, dt, w, z);
    localJ->Add(dt * dt, H->GetLocalGradient(z));
    Jacobian = M->ParallelAssemble(localJ);
    delete localJ;
    HypreParMatrix* Je = Jacobian->EliminateRowsCols(ess_tdof_list);
    delete Je;
    return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
    delete Jacobian;
}


HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace& f,
    Array<int>& ess_bdr, double visc,
    double mu, double K)
    : TimeDependentOperator(2 * f.TrueVSize(), 0.0), fespace(f),
    M(&fespace), S(&fespace), H(&fespace),
    viscosity(visc), M_solver(f.GetComm()), newton_solver(f.GetComm()),
    z(height / 2)
{
    const double rel_tol = 1e-8;
    const int skip_zero_entries = 0;

    const double ref_density = 1.0; // density in the reference configuration
    ConstantCoefficient rho0(ref_density);
    M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
    M.Assemble(skip_zero_entries);
    M.Finalize(skip_zero_entries);
    Mmat = M.ParallelAssemble();
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    HypreParMatrix* Me = Mmat->EliminateRowsCols(ess_tdof_list);
    delete Me;

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(rel_tol);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(30);
    M_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(*Mmat);

    model = new NeoHookeanModel(mu, K);
    H.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
    H.SetEssentialTrueDofs(ess_tdof_list);

    ConstantCoefficient visc_coeff(viscosity);
    S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
    S.Assemble(skip_zero_entries);
    S.Finalize(skip_zero_entries);

    reduced_oper = new ReducedSystemOperator(&M, &S, &H, ess_tdof_list);

    HypreSmoother* J_hypreSmoother = new HypreSmoother;
    J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    J_prec = J_hypreSmoother;

    MINRESSolver* J_minres = new MINRESSolver(f.GetComm());
    J_minres->SetRelTol(rel_tol);
    J_minres->SetAbsTol(0.0);
    J_minres->SetMaxIter(300);
    J_minres->SetPrintLevel(-1);
    J_minres->SetPreconditioner(*J_prec);
    J_solver = J_minres;

    newton_solver.iterative_mode = false;
    newton_solver.SetSolver(*J_solver);
    newton_solver.SetOperator(*reduced_oper);
    newton_solver.SetPrintLevel(1); // print Newton iterations
    newton_solver.SetRelTol(rel_tol);
    newton_solver.SetAbsTol(0.0);
    newton_solver.SetAdaptiveLinRtol(2, 0.5, 0.9);
    newton_solver.SetMaxIter(10);

    dvxdt_prev = 0.0;
}

void HyperelasticOperator::Mult(const Vector& vx, Vector& dvx_dt) const
{
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    int sc = height / 2;
    Vector v(vx.GetData() + 0, sc);
    Vector x(vx.GetData() + sc, sc);
    Vector dv_dt(dvx_dt.GetData() + 0, sc);
    Vector dx_dt(dvx_dt.GetData() + sc, sc);

    H.Mult(x, z);
    if (viscosity != 0.0)
    {
        S.TrueAddMult(v, z);
        z.SetSubVector(ess_tdof_list, 0.0);
    }
    z.Neg(); // z = -z
    M_solver.Mult(z, dv_dt);

    dx_dt = v;
}

void HyperelasticOperator::ImplicitSolve(const double dt,
    const Vector& vx, Vector& dvx_dt)
{
    int sc = height / 2;
    Vector v(vx.GetData() + 0, sc);
    Vector x(vx.GetData() + sc, sc);
    Vector dv_dt(dvx_dt.GetData() + 0, sc);
    Vector dx_dt(dvx_dt.GetData() + sc, sc);

    // By eliminating kx from the coupled system:
    //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
    //    kx = v + dt*kv
    // we reduce it to a nonlinear equation for kv, represented by the
    // reduced_oper. This equation is solved with the newton_solver
    // object (using J_solver and J_prec internally).
    reduced_oper->SetParameters(dt, &v, &x);
    Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
    newton_solver.Mult(zero, dv_dt);
    MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
    add(v, dt, dv_dt, dx_dt);
    dvxdt_prev = dvx_dt;
}

double HyperelasticOperator::ElasticEnergy(const ParGridFunction& x) const
{
    return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction& v) const
{
    double loc_energy = 0.5 * M.InnerProduct(v, v);
    double energy;
    MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
        fespace.GetComm());
    return energy;
}

void HyperelasticOperator::GetElasticEnergyDensity(
    const ParGridFunction& x, ParGridFunction& w) const
{
    ElasticEnergyCoefficient w_coeff(*model, x);
    w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
    delete J_solver;
    delete J_prec;
    delete reduced_oper;
    delete model;
    delete Mmat;
}


double ElasticEnergyCoefficient::Eval(ElementTransformation& T,
    const IntegrationPoint& ip)
{
    model.SetTransformation(T);
    x.GetVectorGradient(T, J);
    // return model.EvalW(J);  // in reference configuration
    return model.EvalW(J) / J.Det(); // in deformed configuration
}


void InitialDeformation(const Vector& x, Vector& y)
{
    // set the initial configuration to be the same as the reference, stress
    // free, configuration
    y = x;
}

void InitialVelocity(const Vector& x, Vector& v)
{
    const int dim = x.Size();
    const double s = 0.1 / 64.;

    v = 0.0;
    v(dim - 1) = s * x(0) * x(0) * (8.0 - x(0));
    v(0) = -s * x(0) * x(0);
}











RomOperator::RomOperator(NonlinearDiffusionOperator* fom_,
    NonlinearDiffusionOperator* fomSp_, const int rrdim_, const int rwdim_,
    const int nldim_, CAROM::SampleMeshManager* smm_,
    const CAROM::Matrix* V_R_, const CAROM::Matrix* U_R_, const CAROM::Matrix* V_W_,
    const CAROM::Matrix* Bsinv,
    const double newton_rel_tol, const double newton_abs_tol, const int newton_iter,
    const CAROM::Matrix* S_, const CAROM::Matrix* Ssinv_,
    const int myid, const bool hyperreduce_source_, const bool oversampling_)
    : TimeDependentOperator(rrdim_ + rwdim_, 0.0),
    newton_solver(),
    fom(fom_), fomSp(fomSp_), BR(NULL), rrdim(rrdim_), rwdim(rwdim_), nldim(nldim_),
    smm(smm_),
    nsamp_R(smm_->GetNumVarSamples("V")),
    nsamp_S(hyperreduce_source_ ? smm_->GetNumVarSamples("S") : 0),
    V_R(*V_R_), U_R(U_R_), V_W(*V_W_), VTU_R(rrdim_, nldim_, false),
    y0(height), dydt_prev(height), zY(nldim, false), zN(std::max(nsamp_R, 1),
        false),
    Vsinv(Bsinv), J(height),
    zS(std::max(nsamp_S, 1), false), zT(std::max(nsamp_S, 1), false), Ssinv(Ssinv_),
    VTCS_W(rwdim, std::max(nsamp_S, 1), false), S(S_),
    VtzR(rrdim_, false), hyperreduce_source(hyperreduce_source_), oversampling(oversampling_)
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

        pfom_R_librom = new CAROM::Vector(pfom_R->GetData(), pfom_R->Size(), false,
            false);
        pfom_W_librom = new CAROM::Vector(pfom_W->GetData(), pfom_W->Size(), false,
            false);

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

void RomOperator::Mult_Hyperreduced(const Vector& dy_dt, Vector& res) const
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
    if (oversampling)
    {
        Vsinv->transposeMult(zN, zY);
    }
    else
    {
        Vsinv->mult(zN, zY);
    }

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

            if (oversampling)
            {
                Ssinv->transposeMult(zT, zS);
            }
            else
            {
                Ssinv->mult(zT, zS);
            }

            // Multiply by the f-basis, followed by C, followed by V_W^T. This is stored in VTCS_W = V_W^T CS.
            VTCS_W.multPlus(resW_librom, zS, -1.0);
        }
        else
        {
            fomSp->Cmat->Mult(fomSp->zW, *psp_W);

            const int nRsp = fomSp->zR.Size();
            const int nWsp = fomSp->zW.Size();
            for (int i = 0; i < rwdim; ++i)
                for (int j = 0; j < nWsp; ++j)
                    res[rrdim + i] -= (*BWsp)(j, i) * (*psp_W)[j];
        }
    }
}

void RomOperator::Mult_FullOrder(const Vector& dy_dt, Vector& res) const
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

void RomOperator::Mult(const Vector& dy_dt, Vector& res) const
{
    if (hyperreduce)
        Mult_Hyperreduced(dy_dt, res);
    else
        Mult_FullOrder(dy_dt, res);
}

void RomOperator::ImplicitSolve(const double dt, const Vector& y, Vector& dy_dt)
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











Operator& RomOperator::GetGradient(const Vector& p) const
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

    for (int i = 0; i < rrdim; ++i)
    {
        if (hyperreduce)
        {
            // Compute the i-th column of M(a'(Pst V_W yW)) Pst V_R.
            for (int j = 0; j < psp_R->Size(); ++j)
                (*psp_R)[j] = (*BRsp)(j, i);

            fomSp->Mprimemat.Mult(*psp_R, zR);

            smm->GetSampledValues("V", zR, z);

            // Note that it would be better to just store VTU_R * Vsinv, but these are small matrices.

            if (oversampling)
            {
                Vsinv->transposeMult(z, r);
            }
            else
            {
                Vsinv->mult(z, r);
            }

            VTU_R.mult(r, c);
        }
        else
        {
            // Compute the i-th column of V_R^T M(a'(V_W yW)) V_R.
            for (int j = 0; j < pfom_R->Size(); ++j)
                (*pfom_R)[j] = V_R(j, i);

            fom->Mprimemat.Mult(*pfom_R, zfomR);
            V_R.transposeMult(*zfomR_librom, c);  // V_R^T M(a'(V_W yW)) V_R(:,i)
        }

        for (int j = 0; j < rrdim; ++j)
            J(j, i) = c(
                j);  // This already includes a factor of current_dt, from Mprimemat.

        for (int j = 0; j < rwdim; ++j)
        {
            J(rrdim + j, i) = -current_dt * (*BR)(j, i);
            J(i, rrdim + j) = current_dt * (*BR)(j, i);
        }
    }

    for (int i = 0; i < rwdim; ++i)
    {
        for (int j = 0; j < rwdim; ++j)
        {
            J(rrdim + j, rrdim + i) = (*CR)(j, i);
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

