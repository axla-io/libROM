
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


typedef enum {FSPACE } FESPACE;

using namespace std;
using namespace mfem;

class ReducedSystemOperator;



class HyperelasticOperator : public TimeDependentOperator
{
    
protected:
    ParBilinearForm *M, *S;
    

    
    CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
    HypreSmoother M_prec; // Preconditioner for the mass matrix M

    

public:
    //HyperelasticOperator(ParFiniteElementSpace& f, Array<int>& ess_bdr,
    HyperelasticOperator(ParFiniteElementSpace& f, Array<int>& ess_tdof_list_,
        double visc, double mu, double K);

    /// Compute the right-hand side of the ODE system.
    virtual void Mult(const Vector& vx, Vector& dvx_dt) const;


    double ElasticEnergy(const ParGridFunction& x) const;
    double KineticEnergy(const ParGridFunction& v) const;
    void GetElasticEnergyDensity(const ParGridFunction& x,
        ParGridFunction& w) const;

    void GetH_dvxdt(const Vector& vx, Vector& dvx_dt, Vector& H);

    ParFiniteElementSpace &fespace;
    double viscosity;
    Array<int> ess_tdof_list;
    ParNonlinearForm* H;
    HyperelasticModel* model;
    mutable Vector z; // auxiliary vector
    mutable Vector z2; // auxiliary vector
    HypreParMatrix* Mmat; // Mass matrix from ParallelAssemble()
    HypreParMatrix Smat; 

    virtual ~HyperelasticOperator();
};




class RomOperator : public TimeDependentOperator
{
private:
    int rxdim, rvdim, hdim;
    int nsamp_H;
    double current_dt;
    bool oversampling;
    CAROM::Matrix* V_v_sp, * V_x_sp, * U_H_sp;
    CAROM::Matrix* V_v_sp_dist;
    CAROM::Vector* psp_librom, * psp_x_librom, * psp_v_librom;
    Vector* psp;
    Vector* psp_x;
    Vector* psp_v;
    mutable Vector zH;
    mutable CAROM::Vector zX;
    mutable CAROM::Vector zN;
    const CAROM::Matrix* Hsinv;
    mutable CAROM::Vector* z_librom;
    mutable Vector z;
    mutable Vector z_x;
    mutable Vector z_v;

    bool hyperreduce;

    CAROM::Vector* pfom_librom, * pfom_x_librom, * pfom_v_librom;
    Vector* pfom;
    Vector* pfom_x;
    Vector* pfom_v;
    mutable Vector* zfom_x;
    mutable Vector* zfom_v;
    CAROM::Vector* zfom_x_librom;

    CAROM::SampleMeshManager* smm;

    CAROM::Vector* z_v_librom;
    CAROM::Vector* z_x_librom;
    


protected:
    CAROM::Matrix* S_hat;
    CAROM::Vector* S_hat_v0;
    Vector* S_hat_v0_temp;
    CAROM::Vector* S_hat_v0_temp_librom;
    CAROM::Matrix* M_hat;
    CAROM::Matrix* M_hat_inv;
    
    const CAROM::Matrix* U_H;
    
    
    HyperelasticOperator* fomSp;

    CGSolver M_hat_solver;    // Krylov solver for inverting the reduced mass matrix M_hat
    HypreSmoother M_hat_prec; // Preconditioner for the reduced mass matrix M_hat

public:
    HyperelasticOperator* fom;


    RomOperator(HyperelasticOperator* fom_,
    HyperelasticOperator* fomSp_, const int rvdim_, const int rxdim_, const int hdim_,
    CAROM::SampleMeshManager* smm_, const Vector v0_, const Vector x0_, const Vector v0_fom_,
    const CAROM::Matrix* V_v_, const CAROM::Matrix* V_x_, const CAROM::Matrix* U_H_,
    const CAROM::Matrix* Hsinv_,
    const int myid, const bool oversampling_);

    virtual void Mult(const Vector& y, Vector& dy_dt) const;
    void Mult_Hyperreduced(const Vector& y, Vector& dy_dt) const;
    void Mult_FullOrder(const Vector& y, Vector& dy_dt) const;

    void Compute_CtAB(const HypreParMatrix* A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix* CtAB);

    CAROM::Matrix V_v, V_x, V_vTU_H;
    Vector x0, v0, v0_fom;

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



// TDO: move this to the library?
CAROM::Matrix* GetFirstColumns(const int N, const CAROM::Matrix* A)
{
    CAROM::Matrix* S = new CAROM::Matrix(A->numRows(), std::min(N, A->numColumns()),
        A->distributed());
    for (int i = 0; i < S->numRows(); ++i)
    {
        for (int j = 0; j < S->numColumns(); ++j)
            (*S)(i, j) = (*A)(i, j);
    }

    // delete A;  // TDO: find a good solution for this.
    return S;
}

// TDO: move this to the library?
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
    int ode_solver_type = 14;
    double t_final = 15.0; // 40.0 For debugging purposes
    double dt = 0.03; // 0.03
    double visc = 1e-2;
    double mu = 0.25;
    double K = 5.0;
    bool adaptive_lin_rtol = true;
    bool visualization = true;
    bool visit = false;
    int vis_steps = 50; // Debug, for normal use it's supposed to be 1

    // ROM parameters
    bool offline = false; // debug mode
    bool merge = false;
    bool online = false;
    bool use_sopt = false;
    int num_samples_req = -1; // 1170 for comparison

    int nsets = 0;

    int id_param = 0;


    // number of basis vectors to use
    int rxdim = -1;
    int rvdim = -1;
    int hdim = -1;

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
    args.AddOption(&rxdim, "-rxdim", "--rxdim",
        "Basis dimension for displacement solution space.");
    args.AddOption(&rvdim, "-rvdim", "--rvdim",
        "Basis dimension for velocity solution space.");
    args.AddOption(&hdim, "-hdim", "--hdim",
        "Basis dimension for the nonlinear term.");
    args.AddOption(&id_param, "-id", "--id", "Parametric index");

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
        // To be added...

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
        // To be added...
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

        // Merge bases
        MergeBasis(true_size, nsets, max_num_snapshots, "V");
        MergeBasis(true_size, nsets, max_num_snapshots, "X");
        MergeBasis(true_size, nsets, max_num_snapshots, "H");


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

    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list); 


    // Store initial vx
    BlockVector vx0 = BlockVector(vx);
    BlockVector vx_diff = BlockVector(vx);
    BlockVector vx_rec = BlockVector(vx);



    Vector* wMFEM = 0;
    CAROM::Vector* w = 0;
    CAROM::Vector* w_v = 0;
    CAROM::Vector* w_x = 0;
    Vector * H_t = new Vector(true_size);

    CAROM::Vector* v_W_librom = 0;
    CAROM::Vector* x_W_librom = 0;
    

    // NOTE: Likely problems here...
    Vector * v_W = new Vector(v_gf.GetTrueVector());
    Vector * x_W = new Vector(x_gf.GetTrueVector());

    //v.SetDataAndSize(&((*v_librom)(0)), true_size);
    v_W_librom = new CAROM::Vector(v_W->GetData(), v_W->Size(), true, false);
    x_W_librom = new CAROM::Vector(x_W->GetData(), x_W->Size(), true, false);

    // For reconstruction
    Vector v_rec(*v_W);
    Vector x_rec(*x_W);

    // 9. Initialize the hyperelastic operator, the GLVis visualization and print
    //    the initial energies.
    HyperelasticOperator oper(fespace, ess_tdof_list, visc, mu, K);
    HyperelasticOperator* soper = 0;

    // Fill dvdt and dxdt
    Vector dvxdt(true_size * 2);
    oper.GetH_dvxdt(vx, dvxdt, *H_t);
    Vector dvdt(dvxdt.GetData() + 0, true_size);
    Vector dxdt(dvxdt.GetData() + true_size, true_size);

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
    CAROM::BasisGenerator* basis_generator_v = 0;  
    CAROM::BasisGenerator* basis_generator_x = 0;
    CAROM::BasisGenerator* basis_generator_H = 0; 


    if (offline) {
        CAROM::Options options(fespace.GetTrueVSize(), max_num_snapshots, 1,
                                 update_right_SV);

            basis_generator_v = new CAROM::BasisGenerator(options, isIncremental,
                basisFileName + "_V");

            basis_generator_x = new CAROM::BasisGenerator(options, isIncremental,
                basisFileName + "_X");

            basis_generator_H = new CAROM::BasisGenerator(options, isIncremental,
                basisFileName + "_H");

    }


    RomOperator* romop = 0;

    const CAROM::Matrix* BV_librom = 0;
    const CAROM::Matrix* BX_librom = 0;
    const CAROM::Matrix* H_librom = 0;
    const CAROM::Matrix* Hsinv = 0;

    int nsamp_H = -1;

    CAROM::SampleMeshManager* smm = nullptr;


    // The online phase
    if (online)
    {
        // Read bases
        CAROM::BasisReader readerV("basisV");
        BV_librom = readerV.getSpatialBasis(0.0);

        
        if (rvdim == -1) // Change rvdim
            rvdim = BV_librom->numColumns();
        else
            BV_librom = GetFirstColumns(rvdim,
                BV_librom);

        MFEM_VERIFY(BV_librom->numRows() == true_size, ""); 

        if (myid == 0)
            printf("reduced V dim = %d\n", rvdim);


        CAROM::BasisReader readerX("basisX");
        BX_librom = readerX.getSpatialBasis(0.0);

        if (rxdim == -1) // Change rxdim
            rxdim = BX_librom->numColumns();
        else
            BX_librom = GetFirstColumns(rxdim,
                BX_librom);  

        MFEM_VERIFY(BX_librom->numRows() == true_size, ""); 

        if (myid == 0)
            printf("reduced X dim = %d\n", rxdim);



        // Hyper reduce H
        CAROM::BasisReader readerH("basisH");
        H_librom = readerH.getSpatialBasis(0.0);

        // Compute sample points using DEIM

        if (hdim == -1) 
        {
            hdim = H_librom->numColumns();
        }

        MFEM_VERIFY(H_librom->numColumns() >= hdim, "");
        //MFEM_VERIFY(FR_librom->numRows() == N1 && FR_librom->numColumns() >= nldim, "");

        if (H_librom->numColumns() > hdim)
            H_librom = GetFirstColumns(hdim, H_librom);

        if (myid == 0)
            printf("reduced H dim = %d\n", hdim);

        vector<int> num_sample_dofs_per_proc(num_procs);

        // Now execute the DEIM algorithm to get the sampling information.
        if (num_samples_req != -1)
        {
            nsamp_H = num_samples_req;
        }
        else
        {
            nsamp_H = hdim;
        }

        CAROM::Matrix* Hsinv = new CAROM::Matrix(nsamp_H, hdim, false);
        vector<int> sample_dofs(nsamp_H);
        if (use_sopt)
        {
            CAROM::S_OPT(H_librom,
                hdim,
                sample_dofs,
                num_sample_dofs_per_proc,
                *Hsinv,
                myid,
                num_procs,
                nsamp_H);
        }
        else if (nsamp_H != hdim)
        {
            CAROM::GNAT(H_librom,
                hdim,
                sample_dofs,
                num_sample_dofs_per_proc,
                *Hsinv,
                myid,
                num_procs,
                nsamp_H);
        }
        else
        {
            CAROM::DEIM(H_librom,
                hdim,
                sample_dofs,
                num_sample_dofs_per_proc,
                *Hsinv,
                myid,
                num_procs);
        }
        

        // Construct sample mesh
        const int nspaces = 1;
        std::vector<ParFiniteElementSpace*> spfespace(nspaces);
        spfespace[0] = &fespace;

        ParFiniteElementSpace* sp_XV_space;

        smm = new CAROM::SampleMeshManager(spfespace);

        vector<int> sample_dofs_empty;
        vector<int> num_sample_dofs_per_proc_empty;
        num_sample_dofs_per_proc_empty.assign(num_procs, 0);

        // smm->RegisterSampledVariable("V", FSPACE, sample_dofs,
        //     num_sample_dofs_per_proc); // NOTE: Probably not needed
        // smm->RegisterSampledVariable("X", FSPACE, sample_dofs,
        //     num_sample_dofs_per_proc);
        // smm->RegisterSampledVariable("H", FSPACE, sample_dofs,
        //     num_sample_dofs_per_proc); // NOTE: Probably not needed
         
        smm->RegisterSampledVariable("V", 0, sample_dofs,
            num_sample_dofs_per_proc); 
        smm->RegisterSampledVariable("X", 0, sample_dofs,
            num_sample_dofs_per_proc);
        smm->RegisterSampledVariable("H", 0, sample_dofs,
            num_sample_dofs_per_proc); 

        smm->ConstructSampleMesh();

        w = new CAROM::Vector(rxdim + rvdim, false);
        w_v = new CAROM::Vector(rvdim, false);
        w_x = new CAROM::Vector(rxdim, false);

        // Initialize w = B_W^T vx.
        BV_librom->transposeMult(*v_W_librom, *w_v);
        BX_librom->transposeMult(*x_W_librom, *w_x);

        
        for (int i = 0; i < rvdim; ++i)
            (*w)(i) = (*w_v)(i);

        for (int i = 0; i < rxdim; ++i)
            (*w)(rvdim + i) = (*w_x)(i);

        *w = 0.0;

        // Note that some of this could be done only on the ROM solver process, but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = new Vector(&((*w)(0)), rxdim + rvdim); 


        // Initial condition hack
        Vector*  w_v0 = 0;
        Vector*  w_x0 = 0;
        
       
        if (myid == 0)
        {
            //sp_XV_space = smm->GetSampleFESpace(FSPACE);
            sp_XV_space = smm->GetSampleFESpace(0);

            int sp_size = sp_XV_space->TrueVSize();
            Array<int> sp_offset(3);
            sp_offset[0] = 0;
            sp_offset[1] = sp_size;
            sp_offset[2] = 2 * sp_size;

            // Initialize sp_p with initial conditions.
            BlockVector sp_vx(sp_offset); 
            ParGridFunction sp_v_gf, sp_x_gf;
            //{
                // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
                //    boundary conditions on a beam-like mesh (see description above).
                
                sp_v_gf.MakeTRef(sp_XV_space, sp_vx, sp_offset[0]); // Associate a new FiniteElementSpace and new true-dof data with the GridFunction.
                sp_x_gf.MakeTRef(sp_XV_space, sp_vx, sp_offset[1]);


                VectorFunctionCoefficient velo(dim, InitialVelocity);
                sp_v_gf.ProjectCoefficient(velo);
                sp_v_gf.SetTrueVector();
                VectorFunctionCoefficient deform(dim, InitialDeformation);
                sp_x_gf.ProjectCoefficient(deform);
                sp_x_gf.SetTrueVector();

                sp_v_gf.SetFromTrueVector();
                sp_x_gf.SetFromTrueVector();
            //}

             // Get initial conditions
            //Vector w_v0(wMFEM->GetData() + 0, rvdim);
            //Vector w_x0(wMFEM->GetData() + rvdim, rxdim);
            w_v0 = new Vector(sp_v_gf.GetTrueVector());
            w_x0 = new Vector(sp_x_gf.GetTrueVector());
            //*w_v0 = 0.0;
            //*w_x0 = 0.0;

            

            /*
           // Convert essential boundary list to binary list
           // Read all boundary vertices in mesh, set 1 if attribute is 1, 0 otherwise

           // Get original mesh
           Mesh* mesh_read = fespace.GetMesh();

            // Get sample mesh
           Mesh* mesh_sp = sp_XV_space->GetMesh();


            // Loop through mesh
            Element* bdr_elem = 0;
            Element* bdr_elem_sp = 0;
            Array<int> elem_vtx(2);
            Array<int> elem_vtx_sp(2);
            cout<<"Number of boundary elems, original mesh: "<<mesh_read->GetNBE()<< '\n';
            int attr = 0;
            
            // For each element int main mesh
           for (size_t i = 0; i < mesh_read->GetNBE(); i++)
           {
            // Get element
            bdr_elem = mesh_read->GetBdrElement(i);

            // Get attribute
            attr = bdr_elem->GetAttribute();

            // If attribute is essential
            if (attr == 1)
            {
                // Get element vertices
                bdr_elem->GetVertices(elem_vtx);

                    // For each element in sample mesh
                    for (size_t j = 0; j < mesh_read->GetNBE(); j++)
                    {
                        // Get sample element
                        bdr_elem_sp = mesh_sp->GetBdrElement(j);

                        // Get sample element vertices
                        bdr_elem_sp->GetVertices(elem_vtx_sp);

                        // If vertices are the same
                        if ((elem_vtx_sp[0] == elem_vtx[0] && elem_vtx_sp[1] == elem_vtx[1])||(elem_vtx_sp[1] == elem_vtx[0] && elem_vtx_sp[0] == elem_vtx[1]))
                        {
                            // Set attribute to be essential
                            bdr_elem_sp->SetAttribute(2);
                        }
                        
                    }
   
            }
            */

           /*  cout<<"Main Elem "<< i << " vertex 1: " << elem_vtx[0] << " vertex 2: "<< elem_vtx[1]<< '\n';
            cout<<"Sample Elem "<< i << " vertex 1: " << elem_vtx_sp[0] << " vertex 2: "<< elem_vtx_sp[1]<< '\n';
            cout<< '\n'; */
            /*
            attr = bdr_elem->GetAttribute();
            //cout<<"Main Elem "<< i << " attribute: " << attr << '\n';
            //cout<<"Sample Elem original "<< i << " attribute: " << bdr_elem_sp->GetAttribute()<< '\n';

            
            
            //bdr_elem_sp->SetAttribute(attr);
            cout<<"Sample Elem new "<< i << " attribute: " << bdr_elem_sp->GetAttribute()<< '\n';
            cout<< '\n';
            */
          // }

           
           
           //mesh_sp->SetAttributes();
            

           // Breakpoint here somewhere
           


           // Loop through binary list and set boundary attribute to 1 if in, else 0.
           // Perform step inbetween (converting binary essential boundary list to sample space)
           cout<<"Binary list"<< '\n';
            CAROM::Matrix Ess_mat(true_size, 1, true);
            for (size_t i = 0; i < true_size; i++)
            {
                Ess_mat(i,0) = 0;
                for (size_t j = 0; j < ess_tdof_list.Size(); j++)
                {
                    if (ess_tdof_list[j] == i )
                    {
                        Ess_mat(i,0) = 1;
                    }
                    
                }

                //cout<<"i: "<< i << " attribute: " << Ess_mat(i,0)<< '\n';
            }

            CAROM::Matrix Ess_mat_sp(sp_size, 1, false);

            // b here

            smm->GatherDistributedMatrixRows("X", Ess_mat, 1, Ess_mat_sp);

            /* cout<<"Original list"<< '\n';
            for (size_t j = 0; j < ess_tdof_list.Size(); j++)
                {
                    cout<<ess_tdof_list[j]<< '\n';
                    
                } */

            // Count number of true elements in new matrix
            int num_ess_sp = 0;

            for (size_t i = 0; i < sp_size; i++)
            {
                if (Ess_mat_sp(i,0) == 1)
                {
                    num_ess_sp += 1;
                }
                
            }
            

            // Initialize list
            Array<int> ess_tdof_list_sp(num_ess_sp);


            // Add indices to list
            int ctr = 0;
            for (size_t i = 0; i < sp_size; i++)
            {
                if (Ess_mat_sp(i,0) == 1)
                {
                    ess_tdof_list_sp[ctr] = i;
                    ctr += 1;
                }
                
            }
            

        // Set essential conditions
        /*
        Array<int> ess_bdr_sp(sp_XV_space->GetMesh()->bdr_attributes.Max());
        ess_bdr_sp = 0;
        //ess_bdr_sp[0] = 1; // boundary attribute 1 (index 0) is fixed
        ess_bdr_sp[1] = 1; // boundary attribute 2 (index 1) is fixed

        */

        // Define operator
        //soper = new HyperelasticOperator(*sp_XV_space, ess_bdr_sp, visc, mu, K);
        soper = new HyperelasticOperator(*sp_XV_space, ess_tdof_list_sp, visc, mu, K); 

        }

        bool hyperreduce = false; // debug

        if (hyperreduce)
        {   // Change class
            romop = new RomOperator(&oper, soper, rvdim, rxdim, hdim, smm, 
            *w_v0, *w_x0, vx0.GetBlock(0),
            //romop = new RomOperator(&oper, &oper, rvdim, rxdim, hdim, smm,
            //vx0.GetBlock(0), vx0.GetBlock(1), vx0.GetBlock(0),
            BV_librom, BX_librom, H_librom, 
            Hsinv, myid, num_samples_req != -1); 
            //Hsinv, myid, true); 
        }
        else
        {
                      
            // Change class
            romop = new RomOperator(&oper, soper, rvdim, rxdim, hdim, smm, 
            vx0.GetBlock(0), vx0.GetBlock(1), vx0.GetBlock(0),
            BV_librom, BX_librom, H_librom, 
            Hsinv, myid, num_samples_req != -1); 

        }
        
        // REMOVE?
        *v_W = 0.0;
        *x_W = 0.0;

            // Display lifted initial energies
            BroadcastUndistributedRomVector(w);

            for (int i=0; i<rvdim; ++i)
                (*w_v)(i) = (*w)(i);
                
            for (int i=0; i<rxdim; ++i)
                (*w_x)(i) = (*w)(rvdim + i);


           romop->V_v.mult(*w_v, *v_W_librom); 
           romop->V_x.mult(*w_x, *x_W_librom); 

            v_rec = *v_W;
            x_rec = *x_W;

           v_rec += vx0.GetBlock(0);
           x_rec += vx0.GetBlock(1);

            v_gf.SetFromTrueDofs(v_rec);
            x_gf.SetFromTrueDofs(x_rec);

            double ee = oper.ElasticEnergy(x_gf);
            double ke = oper.KineticEnergy(v_gf);

            if (myid == 0)
            {
                

                
                cout << "Lifted initial energies, EE = " << ee
                    << ", KE = " << ke << ", ΔTE = " << (ee + ke) - (ee0 + ke0) << endl;


            }
                



        ode_solver->Init(*romop); 

       
        //delete readerV;
        //delete readerX;
        //delete readerH;
    }
    else  
    {
        // fom
        ode_solver->Init(oper); 
    }
    

    // 11. Perform time-integration
    //     (looping over the time iterations, ti, with a time-step dt).
    //     (taking samples and storing it into the pROM object)
    StopWatch fom_timer;

    double t = 0.0;
    vector<double> ts;
    oper.SetTime(t);



    bool last_step = false;
    for (int ti = 1; !last_step; ti++)
    {
        double dt_real = min(dt, t_final - t);

        if (online)
        {   
            if (myid == 0)
            {
                //cout << wMFEM->Elem(10) << ' ' << wMFEM->Elem(60) << '\n';
                // TODO: Add timer here
                ode_solver->Step(*wMFEM, t, dt_real);

                //cout << wMFEM->Elem(10) << ' ' << wMFEM->Elem(60) << '\n';
            }

            

            MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        }
        else
        {
            fom_timer.Start();

            
            ode_solver->Step(vx, t, dt_real);
            

            fom_timer.Stop();

        }

        last_step = (t >= t_final - 1e-8 * dt);

        if (offline)
        {

            if (basis_generator_v->isNextSample(t) || basis_generator_x->isNextSample(t))
            {
                oper.GetH_dvxdt(vx, dvxdt, *H_t);
                vx_diff = BlockVector(vx);
                vx_diff -= vx0;

            }

            // Take samples
            if (basis_generator_v->isNextSample(t))
            {
                basis_generator_v->takeSample(vx_diff.GetBlock(0) , t, dt);
                basis_generator_v->computeNextSampleTime(vx_diff.GetBlock(0), dxdt.GetData(), t);
                basis_generator_H->takeSample(H_t->GetData(), t, dt);
            }

            if (basis_generator_x->isNextSample(t))
            {
                basis_generator_x->takeSample(vx_diff.GetBlock(1), t, dt);
                basis_generator_x->computeNextSampleTime(vx_diff.GetBlock(1) , dxdt.GetData(), t);
                
            }
        }
        

        if (last_step || (ti % vis_steps) == 0)
        {
            if (online)
        {

            BroadcastUndistributedRomVector(w);

            for (int i=0; i<rvdim; ++i)
                (*w_v)(i) = (*w)(i);
                
            for (int i=0; i<rxdim; ++i)
                (*w_x)(i) = (*w)(rvdim + i);


           romop->V_v.mult(*w_v, *v_W_librom); 
           romop->V_x.mult(*w_x, *x_W_librom); 

            v_rec = *v_W;
            x_rec = *x_W;

            //cout << w->item(10) << ' ' << w->item(100) << '\n';

            v_rec += vx0.GetBlock(0);
           x_rec += vx0.GetBlock(1);

            v_gf.SetFromTrueDofs(v_rec);
            x_gf.SetFromTrueDofs(x_rec);
                
        }
        else
        {
            v_gf.SetFromTrueVector();
            x_gf.SetFromTrueVector();

        }

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

    ostringstream velo_name, pos_name;

    velo_name << "velocity." << setfill('0') << setw(6) << myid;
    pos_name << "position." << setfill('0') << setw(6) << myid;

    if (offline)
    {
        // Sample final solution, to prevent extrapolation in ROM between the last sample and the end of the simulation.
        oper.GetH_dvxdt(vx, dvxdt, *H_t);

        // Take samples
        basis_generator_v->takeSample(vx.GetBlock(0), t, dt);
        basis_generator_H->takeSample(H_t->GetData(), t, dt);

        basis_generator_x->takeSample(vx.GetBlock(1), t, dt);

        basis_generator_v->writeSnapshot();
        basis_generator_H->writeSnapshot();

        basis_generator_x->writeSnapshot();

        // Terminate the sampling and write out information.
        delete basis_generator_v;
        delete basis_generator_x;
        delete basis_generator_H;

//    }
// CHange
//    {

        // 12. Save the displaced mesh, the velocity and elastic energy.
        GridFunction* nodes = &x_gf;
        int owns_nodes = 0;
        pmesh->SwapNodes(nodes, owns_nodes);

        ostringstream mesh_name, ee_name;
        mesh_name << "deformed." << setfill('0') << setw(6) << myid;
        ee_name << "elastic_energy." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);
        pmesh->SwapNodes(nodes, owns_nodes);
        ofstream velo_ofs(velo_name.str().c_str());
        velo_ofs.precision(16);
        //v_gf.Save(velo_ofs);

        Vector v_final(vx.GetBlock(0));
        for (int i = 0; i < v_final.Size(); ++i)
        {
            velo_ofs << v_final[i] << std::endl;
        }

        
        ofstream pos_ofs(pos_name.str().c_str());
        pos_ofs.precision(16);
        //x_gf.Save(pos_ofs);

        Vector x_final(vx.GetBlock(1));
        for (int i = 0; i < x_final.Size(); ++i)
        {
            pos_ofs << x_final[i] << std::endl;
        }

        ofstream ee_ofs(ee_name.str().c_str());
        ee_ofs.precision(8);
        oper.GetElasticEnergyDensity(x_gf, w_gf);
        w_gf.Save(ee_ofs);

    }

//bool test_this = false;
    // 15. Calculate the relative error between the ROM final solution and the true solution.
    //if (test_this)

    if (online)
    {
        // Initialize FOM solution
        Vector v_fom(v_rec.Size());
        Vector x_fom(x_rec.Size());

        ifstream fom_v_file, fom_x_file;

        // Open and load file
        fom_v_file.open(velo_name.str().c_str());
        fom_x_file.open(pos_name.str().c_str());

        v_fom.Load(fom_v_file, v_rec.Size());
        x_fom.Load(fom_x_file, x_rec.Size());

        fom_v_file.close();
        fom_x_file.close();


        
        //for (size_t i = 0; i < x_rec.Size(); i++)
        //{
           /* cout << "X Difference: " << diff_x[i] <<
                    "    X FOM value: " << x_fom[i] << endl;
                    */
            /* cout << "X ROM prediction: " << x_rec[i] <<
                    "    X FOM value: " << x_fom[i] << endl; */

        //}
        

        
        // Get difference vector
        Vector diff_v(v_rec.Size());
        Vector diff_x(x_rec.Size());

        subtract(v_rec, v_fom, diff_v);
        subtract(x_rec, x_fom, diff_x);

        // Get norms
        double tot_diff_norm_v = sqrt(InnerProduct(MPI_COMM_WORLD, diff_v, diff_v));
        double tot_diff_norm_x = sqrt(InnerProduct(MPI_COMM_WORLD, diff_x, diff_x));

        double tot_v_fom_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                            v_fom, v_fom));
        double tot_x_fom_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                            x_fom, x_fom));



        if (myid == 0)
        {
            cout << "Relative error of ROM position (x) at t_final: " << t_final <<
                    " is " << tot_diff_norm_x / tot_x_fom_norm << endl;
            cout << "Relative error of ROM velocity (v) at t_final: " << t_final <<
                    " is " << tot_diff_norm_v / tot_v_fom_norm << endl;
        }
    }
    
    


    // 16. Free the used memory.
    delete ode_solver;
    delete pmesh;
    delete H_t;
    delete v_W;
    delete x_W;

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



HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace& f,
    Array<int>& ess_tdof_list_, double visc,
    double mu, double K)
    : TimeDependentOperator(2 * f.TrueVSize(), 0.0), fespace(f),ess_tdof_list(ess_tdof_list_),
    //M(&fespace), S(&fespace), H(&fespace),
    M(NULL), S(NULL), H(NULL),
    viscosity(visc), M_solver(f.GetComm()),
    z(height / 2), z2(height / 2)
{
    const double rel_tol = 1e-8;
    const int skip_zero_entries = 0;

    const double ref_density = 1.0; // density in the reference configuration
    ConstantCoefficient rho0(ref_density);

    M = new ParBilinearForm(&fespace);
    M->AddDomainIntegrator(new VectorMassIntegrator(rho0));
    M->Assemble(skip_zero_entries);
    M->Finalize(skip_zero_entries);
    Mmat = M->ParallelAssemble();
    //fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list); 
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


    H = new ParNonlinearForm(&fespace);
    model = new NeoHookeanModel(mu, K);
    H->AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
    H->SetEssentialTrueDofs(ess_tdof_list);

    ConstantCoefficient visc_coeff(viscosity);
    S = new ParBilinearForm(&fespace);
    S->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
    S->Assemble(skip_zero_entries);
    S->Finalize(skip_zero_entries);
    //Smat = S->ParallelAssemble();
    S->FormSystemMatrix(ess_tdof_list, Smat);

}

void HyperelasticOperator::Mult(const Vector& vx, Vector& dvx_dt) const
{
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    int sc = height / 2;
    Vector v(vx.GetData() + 0, sc);
    Vector x(vx.GetData() + sc, sc);
    Vector dv_dt(dvx_dt.GetData() + 0, sc);
    Vector dx_dt(dvx_dt.GetData() + sc, sc);

    H->Mult(x, z);

    if (viscosity != 0.0)
    {
        //S->TrueAddMult(v, z);
        //z.SetSubVector(ess_tdof_list, 0.0);
        Smat.Mult(v, z2);
        z += z2;
    }
    z.Neg(); // z = -z
    M_solver.Mult(z, dv_dt);

    dx_dt = v;

}

void HyperelasticOperator::GetH_dvxdt(const Vector& vx, Vector& dvx_dt, Vector& H_new)
{
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    int sc = height / 2;
    Vector v(vx.GetData() + 0, sc);
    Vector x(vx.GetData() + sc, sc);
    Vector dv_dt(dvx_dt.GetData() + 0, sc);
    Vector dx_dt(dvx_dt.GetData() + sc, sc);

    H->Mult(x, z);
    H_new = z; // Store H for sampling

    if (viscosity != 0.0)
    {
        //S->TrueAddMult(v, z);
        //z.SetSubVector(ess_tdof_list, 0.0);
        Smat.Mult(v, z2);
        z += z2;
    }
    z.Neg(); // z = -z
    M_solver.Mult(z, dv_dt);

    dx_dt = v;
}



double HyperelasticOperator::ElasticEnergy(const ParGridFunction& x) const
{
    return H->GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction& v) const
{
    double loc_energy = 0.5 * M->InnerProduct(v, v);
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
    delete model;
    delete Mmat;
    delete M;
    delete S;
    delete H;
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




RomOperator::RomOperator(HyperelasticOperator* fom_,
    HyperelasticOperator* fomSp_, const int rvdim_, const int rxdim_, const int hdim_,
    CAROM::SampleMeshManager* smm_, const Vector v0_, const Vector x0_, const Vector v0_fom_,
    const CAROM::Matrix* V_v_, const CAROM::Matrix* V_x_, const CAROM::Matrix* U_H_,
    const CAROM::Matrix* Hsinv_,
    const int myid, const bool oversampling_)
    : TimeDependentOperator(rxdim_ + rvdim_, 0.0),
    fom(fom_), fomSp(fomSp_), rxdim(rxdim_), rvdim(rvdim_), hdim(hdim_), x0(x0_), v0(v0_), v0_fom(v0_fom_),
    smm(smm_), nsamp_H(smm_->GetNumVarSamples("H")), V_x(*V_x_), V_v(*V_v_), U_H(U_H_), Hsinv(Hsinv_),
    zN(std::max(nsamp_H, 1), false), zX(std::max(nsamp_H, 1), false), M_hat_solver(fom_->fespace.GetComm()),
    oversampling(oversampling_), z(height / 2)
{

    if (myid == 0)
    {
        V_v_sp = new CAROM::Matrix(fomSp->Height() / 2, rvdim, false);
        V_x_sp = new CAROM::Matrix(fomSp->Height() / 2, rxdim, false);
        U_H_sp = new CAROM::Matrix(fomSp->Height() / 2, hdim, false);
    }


    hyperreduce = false; // Debug mode


    // Gather distributed vectors
    smm->GatherDistributedMatrixRows("V", V_v, rvdim, *V_v_sp);
    smm->GatherDistributedMatrixRows("X", V_x, rxdim, *V_x_sp);
    smm->GatherDistributedMatrixRows("H", *U_H, hdim, *U_H_sp);


    // Create V_vTU_H, for hyperreduction
    V_v.transposeMult(*U_H, V_vTU_H);
    //V_v_sp->transposeMult(*U_H_sp, V_vTU_H);

    S_hat = new CAROM::Matrix(rvdim, rvdim, false);
    S_hat_v0 = new CAROM::Vector(rvdim, false);
    S_hat_v0_temp = new Vector(v0_fom.Size());
    S_hat_v0_temp_librom = new CAROM::Vector(S_hat_v0_temp->GetData(), S_hat_v0_temp->Size(), true, false);
    M_hat = new CAROM::Matrix(rvdim, rvdim, false);
    M_hat_inv = new CAROM::Matrix(rvdim, rvdim, false);
    
    // Create S_hat
    Compute_CtAB(&(fom->Smat), V_v, V_v, S_hat); 

    // Apply S_hat to the initial velocity and store
    fom->Smat.Mult(v0_fom, *S_hat_v0_temp);
    V_v.transposeMult(*S_hat_v0_temp_librom, S_hat_v0); 

    // Create M_hat
    Compute_CtAB(fom->Mmat, V_v, V_v, M_hat);

    // Invert M_hat and store
    M_hat->inverse(*M_hat_inv);

    //if (myid == 0)
    if (hyperreduce)
    {
        const int spdim = fomSp->Height();  // Reduced height

        zH.SetSize(spdim / 2); // Samples of H

        // Allocate auxillary vectors
        //z_x_librom = new CAROM::Vector(spdim / 2, false);
        //z_v_librom = new CAROM::Vector(spdim / 2, false);
        //z_librom = new CAROM::Vector(spdim / 2, false);
        //z = new Vector(&((*z_librom)(0)), spdim / 2);
        //z_v = new Vector(&((*z_v_librom)(0)), spdim / 2);
        //z_x = new Vector(&((*z_x_librom)(0)), spdim / 2);

        z.SetSize(spdim / 2);
        z_v.SetSize(spdim / 2);
        z_x.SetSize(spdim / 2);
        z_librom = new CAROM::Vector(z.GetData(), z.Size(), false, false);
        z_v_librom = new CAROM::Vector(z_v.GetData(), z_v.Size(), false, false);
        z_x_librom = new CAROM::Vector(z_x.GetData(), z_x.Size(), false, false);

        // This is for saving the recreated predictions
        psp_librom = new CAROM::Vector(spdim, false);
        psp = new Vector(&((*psp_librom)(0)), spdim);

        // Define sub-vectors of psp.
        psp_x = new Vector(psp->GetData(), spdim / 2);
        psp_v = new Vector(psp->GetData() + spdim / 2, spdim / 2);

        psp_x_librom = new CAROM::Vector(psp_x->GetData(), psp_x->Size(), false, false);
        psp_v_librom = new CAROM::Vector(psp_v->GetData(), psp_v->Size(), false, false);


    }

    if (!hyperreduce)
    {
        const int fdim = fom->Height(); // Unreduced height

        z.SetSize(fdim / 2);
        z_v.SetSize(fdim / 2);
        z_x.SetSize(fdim / 2);
        z_librom = new CAROM::Vector(z.GetData(), z.Size(), false, false);
        z_v_librom = new CAROM::Vector(z_v.GetData(), z_v.Size(), true, false);
        z_x_librom = new CAROM::Vector(z_x.GetData(), z_x.Size(), true, false);

        // This is for saving the recreated predictions
        pfom_librom = new CAROM::Vector(fdim, false);
        pfom = new Vector(&((*pfom_librom)(0)), fdim);

        // Define sub-vectors of pfom.
        pfom_x = new Vector(pfom->GetData(), fdim / 2);
        pfom_v = new Vector(pfom->GetData() + fdim / 2, fdim / 2);
        zfom_x = new Vector(fdim / 2);
        zfom_x_librom = new CAROM::Vector(zfom_x->GetData(), zfom_x->Size(), true, false);


        pfom_x_librom = new CAROM::Vector(pfom_x->GetData(), pfom_x->Size(), true,
            false);
        pfom_v_librom = new CAROM::Vector(pfom_v->GetData(), pfom_v->Size(), true,
            false);
    }

}




RomOperator::~RomOperator()
{
    delete S_hat;
    delete M_hat;
    delete M_hat_inv;
}



void RomOperator::Mult_Hyperreduced(const Vector& vx, Vector& dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    //Vector x(vx.GetData() + rvdim, rxdim);
    CAROM::Vector v_librom(vx.GetData(), rvdim, false, false);
    CAROM::Vector x_librom(vx.GetData() + rvdim, rxdim, false, false);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);
    CAROM::Vector dv_dt_librom(dv_dt.GetData(), rvdim, false, false);
    CAROM::Vector dx_dt_librom(dx_dt.GetData(), rxdim, false, false);

    
    /* cout<<"x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<x_librom.item(i)<<'\n';
    }

    cout<<"v_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<v_librom.item(i)<<'\n';
    }  */
    
    // Lift the x-, and v-vectors
    // I.e. perform v = v0 + V_v v^, where v^ is the input
    V_v_sp->mult(v_librom, *z_v_librom); 
    V_x_sp->mult(x_librom, *z_x_librom);

    /* cout<<"z_v_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_v_librom->item(i)<<'\n';
    }

    cout<<"z_x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_x_librom->item(i)<<'\n';
    }  */

    add(z_v, v0, *psp_v); // Store liftings
    add(z_x, x0, *psp_x); 

   /*  cout<<"psp_v contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<psp_v_librom->item(i)<<'\n';
    }

    cout<<"psp_x contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<psp_x_librom->item(i)<<'\n';
    }  */

    // Hyperreduce H
    // Apply H to x to get zH
    fomSp->H->Mult(*psp_x, zH);
    //fom->H->Mult(*psp_x, zH);

    /* cout<<"zH contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<zH.Elem(i)<<'\n';
    }  */

    // Sample the values from zH
    smm->GetSampledValues("H", zH, zN);

    /* cout<<"zN contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<zN.item(i)<<'\n';
    }  */

    // Apply inverse H-basis
    if (oversampling)
    {
        Hsinv->transposeMult(zN, zX);
    }
    else
    {
        Hsinv->mult(zN, zX);
    }
/* 
    cout<<"zX contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<zX.item(i)<<'\n';
    }  */

    //cout<<"This should be same as zN:"<<'\n';
    //CAROM::Vector test_zX(zX.getData(), zX.dim(), false, true);
    //CAROM::Vector test_zN(zN.getData(), zN.dim(), true, true);
    //U_H->mult(test_zX, test_zN);


   /*  for (size_t i = 0; i < 25; i++)
    {
        cout<<test_zN.item(i)<<'\n';
    } 
 */
    // Multiply by V_v^T * U_H
    V_vTU_H.mult(zX, z_librom);

    /* 
     cout<<"z_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_librom->item(i)<<'\n';
    }  */
    

    if (fomSp->viscosity != 0.0)
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(*z_librom, v_librom, 1.0); 
        *z_librom += *S_hat_v0;
        //z.SetSubVector(fomSp->ess_tdof_list, 0.0);
    }

    z.Neg(); // z = -z, because we are calculating the residual.
    M_hat_inv->mult(*z_librom, dv_dt_librom); // to invert reduced mass matrix operator.

    V_x_sp->transposeMult(*psp_v_librom, dx_dt_librom);

   
    
     
}



void RomOperator::Mult_FullOrder(const Vector& vx, Vector& dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    //Vector x(vx.GetData() + rvdim, rxdim);
    CAROM::Vector v_librom(vx.GetData(), rvdim, false, false);
    CAROM::Vector x_librom(vx.GetData() + rvdim, rxdim, false, false);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);
    CAROM::Vector dv_dt_librom(dv_dt.GetData(), rvdim, false, false);
    CAROM::Vector dx_dt_librom(dx_dt.GetData(), rxdim, false, false);


    
    /* cout<<"x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<x_librom.item(i)<<'\n';
    }

    cout<<"v_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<v_librom.item(i)<<'\n';
    }  */
    

    // Lift the x-, and v-vectors
    // I.e. perform v = v0 + V_v v^, where v^ is the input
    V_x.mult(x_librom, *z_x_librom); 
    V_v.mult(v_librom, *z_v_librom); 

    /* cout<<"z_v_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_v_librom->item(i)<<'\n';
    }

    cout<<"z_x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_x_librom->item(i)<<'\n';
    }  */

    add(z_x, x0, *pfom_x); // Store liftings
    add(z_v, v0, *pfom_v);

    /* cout<<"pfom_v_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<pfom_v_librom->item(i)<<'\n';
    } 

    cout<<"pfom_x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<pfom_x_librom->item(i)<<'\n';
    }  */

    // Apply H to x to get z
     fom->H->Mult(*pfom_x, *zfom_x);
    //fom->H->Mult(x0, *zfom_x);

    /* cout<<"zfom_x_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<zfom_x_librom->item(i)<<'\n';
    }  */

    

    V_v.transposeMult(*zfom_x_librom, z_librom); 

    
    /*  cout<<"z_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<z_librom->item(i)<<'\n';
    } 
     */

    if (fom->viscosity != 0.0) 
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(*z_librom, v_librom, 1.0); 
        *z_librom += *S_hat_v0;
        //z.SetSubVector(fom->ess_tdof_list, 0.0);
    }

    z.Neg(); // z = -z, because we are calculating the residual.
    M_hat_inv->mult(*z_librom, dv_dt_librom); // to invert reduced mass matrix operator.
    

    V_x.transposeMult(*pfom_v_librom, dx_dt_librom);


   
    /* cout<<"dv_dt_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<dv_dt_librom.item(i)<<'\n';
    }
   

   
    cout<<"dx_dt_librom contents:"<<'\n';
    for (size_t i = 0; i < 25; i++)
    {
        cout<<dx_dt_librom.item(i)<<'\n';
    }  */


    /*

    cout<<"dv_dt_librom contents:"<<'\n';
    for (size_t i = 0; i < dv_dt_librom.dim(); i++)
    {
        cout<<dv_dt_librom.item(i)<<'\n';
    }
   

   
    cout<<"dx_dt_librom contents:"<<'\n';
    for (size_t i = 0; i < dx_dt_librom.dim(); i++)
    {
        cout<<dx_dt_librom.item(i)<<'\n';
    } 
    */

    // B here

}

void RomOperator::Mult(const Vector& vx, Vector& dvx_dt) const
{
    if (hyperreduce)
        Mult_Hyperreduced(vx, dvx_dt);
    else
        Mult_FullOrder(vx, dvx_dt);
}


void RomOperator::Compute_CtAB(const HypreParMatrix* A,
    const CAROM::Matrix& B,  // Distributed matrix.
    const CAROM::Matrix& C,  // Distributed matrix.
    CAROM::Matrix*
    CtAB)     // Non-distributed (local) matrix, computed identically and redundantly on every process.
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