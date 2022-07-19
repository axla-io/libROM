
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
    ParBilinearForm M, S;
    ParNonlinearForm H;
    HyperelasticModel* model;

    HypreParMatrix* Mmat; // Mass matrix from ParallelAssemble()
    CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
    HypreSmoother M_prec; // Preconditioner for the mass matrix M

    mutable Vector z; // auxiliary vector

    Vector dvxdt_prev; // for computing sample time
    Vector H_prev; // for sampling H

public:
    HyperelasticOperator(ParFiniteElementSpace& f, Array<int>& ess_bdr,
        double visc, double mu, double K);

    /// Compute the right-hand side of the ODE system.
    virtual void Mult(const Vector& vx, Vector& dvx_dt) const;


    double ElasticEnergy(const ParGridFunction& x) const;
    double KineticEnergy(const ParGridFunction& v) const;
    void GetElasticEnergyDensity(const ParGridFunction& x,
        ParGridFunction& w) const;

    void CopyDvxDt(Vector& dvxdt) const
    {
        dvxdt = dvxdt_prev;
    }

    void CopyH_t(Vector& H_t) const
    {
        H_t = H_prev;
    }

    ParFiniteElementSpace& fespace;
    double viscosity;
    Array<int> ess_tdof_list;

    virtual ~HyperelasticOperator();
};




class RomOperator : public TimeDependentOperator
{
private:
    int rxdim, rvdim;
    int nsamp_H;
    double current_dt;
    bool oversampling;
    CAROM::Matrix* V_v_sp, * V_x_sp;
    CAROM::Vector* psp_librom, * psp_x_librom, * psp_v_librom;
    Vector* psp;
    Vector* psp_x;
    Vector* psp_v;
    mutable Vector zH;
    mutable CAROM::Vector zX;
    mutable CAROM::Vector zN;
    const CAROM::Matrix* Hsinv;
    mutable Vector z;

    bool hyperreduce;

    CAROM::Vector* pfom_librom, * pfom_x_librom, * pfom_v_librom;
    Vector* pfom;
    Vector* pfom_x;
    Vector* pfom_v;
    mutable Vector zfomx;
    mutable Vector zfomv;
    CAROM::Vector* zfomx_librom;

    CAROM::SampleMeshManager* smm;

    mutable Vector z_v_librom;
    mutable Vector z_x_librom;
    


protected:
    CAROM::Matrix* S_hat;
    CAROM::Matrix* M_hat;
    const CAROM::Matrix* U_H;
    Vector x0, v0;
    Vector H_prev;
    HyperelasticOperator* fom;
    HyperelasticOperator* fomSp;

    CGSolver M_hat_solver;    // Krylov solver for inverting the reduced mass matrix M_hat
    HypreSmoother M_hat_prec; // Preconditioner for the reduced mass matrix M_hat

public:
    RomOperator(HyperelasticOperator* fom_,
        HyperelasticOperator* fomSp_, const int rrdim_,
        CAROM::SampleMeshManager* smm_,
        const CAROM::Matrix* V_x_, const CAROM::Matrix* V_v_, const CAROM::Matrix* U_H_,
        const CAROM::Matrix* Hsinv,
        const int myid, const bool oversampling_);

    virtual void Mult(const Vector& y, Vector& dy_dt) const;
    void Mult_Hyperreduced(const Vector& y, Vector& dy_dt) const;
    void Mult_FullOrder(const Vector& y, Vector& dy_dt) const;

    void Compute_CtAB(const HypreParMatrix* A, const CAROM::Matrix& B, const CAROM::Matrix& C, CAROM::Matrix* CtAB);

    CAROM::Matrix V_v, V_x, V_vTU_H;

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

    Vector* wMFEM = 0;
    CAROM::Vector* w = 0;
    CAROM::Vector* w_v = 0;
    CAROM::Vector* w_x = 0;
    Vector H_t = new Vector(true_size * 2);


    
    CAROM::Vector* v_W_librom = 0;
    CAROM::Vector* x_W_librom = 0;
    


    // NOTE: Likely problems here...
    Vector v_W = Vector(v_gf.GetTrueVector());
    Vector x_W = Vector(v_gf.GetTrueVector());

    //v.SetDataAndSize(&((*v_librom)(0)), true_size);
    v_W_librom = new CAROM::Vector(v_W.GetData(), v_W.Size(), true, false);
    x_W_librom = new CAROM::Vector(x_W.GetData(), x_W.Size(), true, false);



    // 9. Initialize the hyperelastic operator, the GLVis visualization and print
    //    the initial energies.
    HyperelasticOperator oper(fespace, ess_bdr, visc, mu, K);
    HyperelasticOperator* soper = 0;

    // Fill dvdt and dxdt
    Vector dvxdt;
    oper.CopyDvxDt(dvxdt);
    int sc = dvxdt.Size() / 2;
    Vector dvdt(dvxdt.GetData() + 0, sc);
    Vector dxdt(dvxdt.GetData() + sc, sc);

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
    CAROM::BasisGenerator* basis_generator_v = 0;  
    CAROM::BasisGenerator* basis_generator_x = 0;
    CAROM::BasisGenerator* basis_generator_H = 0; 


    if (offline) {
        CAROM::Options options_v(true_size, max_num_snapshots, 1,
            update_right_SV);
        CAROM::Options options_x(true_size, max_num_snapshots, 1,
            update_right_SV);

            basis_generator_v = new CAROM::BasisGenerator(options_v, isIncremental,
                basisFileName + "_V");

            basis_generator_x = new CAROM::BasisGenerator(options_x, isIncremental,
                basisFileName + "_X");

            basis_generator_H = new CAROM::BasisGenerator(options_v, isIncremental,
                basisFileName + "_H");

    }


    RomOperator* romop = 0;

    const CAROM::Matrix* BV_librom = 0;
    const CAROM::Matrix* BX_librom = 0;
    const CAROM::Matrix* H_librom = 0;
    const CAROM::Matrix* Hsinv = 0;

    int nsamp_v = -1;
    int nsamp_x = -1;
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
            printf("reduced V dim = %d\n", rxdim);


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

        vector<int> num_sample_dofs_per_proc_empty;
        num_sample_dofs_per_proc_empty.assign(num_procs, 0);


        smm->RegisterSampledVariable("V", FSPACE, sample_dofs,
            num_sample_dofs_per_proc); // NOTE: Probably not needed
        smm->RegisterSampledVariable("X", FSPACE, sample_dofs,
            num_sample_dofs_per_proc);
        smm->RegisterSampledVariable("H", FSPACE, sample_dofs,
            num_sample_dofs_per_proc); // NOTE: Probably not needed
    
        smm->ConstructSampleMesh();

        w = new CAROM::Vector(rxdim + rvdim, false);
        w_v = new CAROM::Vector(rvdim, false);
        w_x = new CAROM::Vector(rxdim, false);

        // Initialize w = B_W^T vx.
        BV_librom->transposeMult(v_W_librom, *w_v);
        BX_librom->transposeMult(x_W_librom, *w_x);

        for (int i = 0; i < rvdim; ++i)
            (*w)(i) = (*w_v)(i);

        for (int i = 0; i < rxdim; ++i)
            (*w)(rvdim + i) = (*w_x)(i);

        // Note that some of this could be done only on the ROM solver process, but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = new Vector(&((*w)(0)), rxdim + rvdim); 

        if (myid == 0)
        {
            sp_XV_space = smm->GetSampleFESpace(FSPACE);

            // Initialize sp_p with initial conditions.
            {
                // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
                //    boundary conditions on a beam-like mesh (see description above).
                BlockVector sp_vx(true_offset);
                ParGridFunction sp_v_gf, sp_x_gf;
                sp_v_gf.MakeTRef(sp_XV_space, sp_vx, true_offset[0]); // Associate a new FiniteElementSpace and new true-dof data with the GridFunction.
                sp_x_gf.MakeTRef(sp_XV_space, sp_vx, true_offset[1]);


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
            soper = new HyperelasticOperator oper(**sp_FEspace, ess_bdr, visc, mu, K);

        }


        romop = new RomOperator(&oper, soper, rxdim, rvdim, hdim, smm,
            BV_librom, BX_librom, H_librom, w_v, w_x,
            Hsinv, myid, num_samples_req != -1); 

        ode_solver->Init(*romop); 

    }
    else  // fom
        ode_solver->Init(oper); 


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
            if (myid == 0)
            {
                ode_solver->Step(*wMFEM, t, dt_real);
            }

            MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        if (offline)
        {

            if (basis_generator_v->isNextSample(t) || basis_generator_x->isNextSample(t))
            {
                oper.CopyDvxDt(dvxdt);

            }

            // Take samples
            if (basis_generator_v->isNextSample(t))
            {
                basis_generator_v->takeSample(vx.GetBlock(0), t, dt);
                basis_generator_v->computeNextSampleTime(vx.GetBlock(0), dxdt.GetData(), t);

                oper.CopyH_t(H_t);
                basis_generator_H->takeSample(H_t.GetData(), t, dt);
            }

            if (basis_generator_x->isNextSample(t))
            {
                basis_generator_x->takeSample(vx.GetBlock(1), t, dt);
                basis_generator_x->computeNextSampleTime(vx.GetBlock(1), dxdt.GetData(), t);

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
        oper.CopyH_t(H_t);

        // Take samples
        basis_generator_v->takeSample(vx.GetBlock(0), t, dt);

        oper.CopyH_t(H_t);
        basis_generator_H->takeSample(H_t.GetData(), t, dt);

        basis_generator_x->takeSample(vx.GetBlock(1), t, dt);

    




        // Terminate the sampling and write out information.
        delete basis_generator_v;
        delete basis_generator_x;
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



HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace& f,
    Array<int>& ess_bdr, double visc,
    double mu, double K)
    : TimeDependentOperator(2 * f.TrueVSize(), 0.0), fespace(f),
    M(&fespace), S(&fespace), H(&fespace),
    viscosity(visc), M_solver(f.GetComm()),
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

    dvxdt_prev = 0.0;
    H_prev = 0.0;
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
    H_prev = z; // Store H for sampling

    if (viscosity != 0.0)
    {
        S.TrueAddMult(v, z);
        z.SetSubVector(ess_tdof_list, 0.0);
    }
    z.Neg(); // z = -z
    M_solver.Mult(z, dv_dt);

    dx_dt = v;
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




RomOperator::RomOperator(HyperelasticOperator* fom_,
    HyperelasticOperator* fomSp_, const int rrdim_,
    CAROM::SampleMeshManager* smm_, const Vector x0_, const Vector v0_,
    const CAROM::Matrix* V_x_, const CAROM::Matrix* V_v_, const CAROM::Matrix* U_H_,
    const CAROM::Matrix* Hsinv,
    const int myid, const bool oversampling_)
    : TimeDependentOperator(rrdim_ + rwdim_, 0.0),
    fom(fom_), fomSp(fomSp_), rxdim(rrdim_), x0(x0_), v0(v0_), 
    smm(smm_), nsamp_H(smm_->GetNumVarSamples("H")), V_x(*V_x_), V_v(*V_v_), U_H(*U_H_), .
    zN(std::max(nsamp_H, 1), false), M_hat_solver(fom_->fespace.GetComm()),
    oversampling(oversampling_), z(height / 2)
{

    if (myid == 0)
    {
        V_v_sp = new CAROM::Matrix(fomSp->Height() / 2, rxdim, false);
        V_x_sp = new CAROM::Matrix(fomSp->Height() / 2, rvdim, false);
    }

    // Gather distributed vectors
    smm->GatherDistributedMatrixRows("V", V_v, rvdim, *V_v_sp);
    smm->GatherDistributedMatrixRows("X", V_x, rxdim, *V_x_sp);

    // Create V_vTU_H, for hyperreduction
    V_v.transposeMult(*U_H, V_vTU_H);

    // Create S_hat
    S_hat = new CAROM::Matrix(rvdim, rvdim, false);
    Compute_CtAB(fom->S, V_v, V_v, S_hat); 

    // Create M_hat
    M_hat = new CAROM::Matrix(rvdim, rvdim, false);
    Compute_CtAB(fom->Mmat, V_v, V_v, M_hat);

    // Create M_hat_solver
    const double rel_tol = 1e-8;
    M_hat_solver.iterative_mode = false;
    M_hat_solver.SetRelTol(rel_tol);
    M_hat_solver.SetAbsTol(0.0);
    M_hat_solver.SetMaxIter(30);
    M_hat_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_hat_solver.SetPreconditioner(M_prec);
    M_hat_solver.SetOperator(*M_hat);


    if (myid == 0)
    {
        const int spdim = fomSp->Height();  // Reduced height
        zH.SetSize(spdim / 2); // Samples of H

        // Set size of H storage vector
        H_prev(spdim / 2);

        // Allocate auxillary vectors
        z_x_librom = new CAROM::Vector(spdim / 2, false);
        z_v_librom = new CAROM::Vector(spdim / 2, false);
        z = new Vector(spdim / 2);

        // This is for saving the recreated predictions
        psp_librom = new CAROM::Vector(spdim, false);
        psp = new Vector(&((*psp_librom)(0)), spdim);

        // Define sub-vectors of psp.
        psp_x = new Vector(psp->GetData(), spdim / 2);
        psp_v = new Vector(psp->GetData() + spdim / 2, spdim / 2);

        psp_x_librom = new CAROM::Vector(psp_x->GetData(), psp_x->Size(), false, false);
        psp_v_librom = new CAROM::Vector(psp_v->GetData(), psp_v->Size(), false, false);


    }

    hyperreduce = true;

    if (!hyperreduce)
    {
        const int fdim = fom->Height(); // Unreduced height

        // This is for saving the recreated predictions
        pfom_librom = new CAROM::Vector(fdim, false);
        pfom = new Vector(&((*pfom_librom)(0)), fdim);

        // Define sub-vectors of pfom.
        
        pfom_x = new Vector(pfom->GetData(), fdim / 2);
        pfom_v = new Vector(pfom->GetData() + fdim / 2, fdim / 2);
        zfomx = new Vector(fdim / 2);


        pfom_x_librom = new CAROM::Vector(pfom_x->GetData(), pfom_x->Size(), false,
            false);
        pfom_v_librom = new CAROM::Vector(pfom_v->GetData(), pfom_v->Size(), false,
            false);
        zfomx_librom = new CAROM::Vector(zfomx->GetData(), zfomx->Size(), false,
            false);
    }

}




RomOperator::~RomOperator()
{
    delete S_hat;
    delete M_hat;
}



void RomOperator::Mult_Hyperreduced(const Vector& vx, Vector& dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    Vector x(vx.GetData() + rvdim, rxdim);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);

    // Lift the x-, and v-vectors
    // I.e. perform v = v0 + V_v v^, where v^ is the input
    V_x_sp->mult(x, *z_x);
    V_v_sp->mult(v, *z_v);

    add(z_x, x0, *psp_x) // Store liftings
    add(z_v, v0, *psp_v)

    // Hyperreduce H
    // Apply H to x to get zH
    fomSp->H->Mult(*psp_x, zH);

    // Sample the values from zH
    smm->GetSampledValues("X", zH, zN);

    // Apply inverse H-basis
    if (oversampling)
    {
        Hsinv->transposeMult(zN, zX);
    }
    else
    {
        Hsinv->mult(zN, zX);
    }

    // Multiply by V_v^T * U_H
    V_vTU_H.mult(zX, z); 


    if (fomSp->viscosity != 0.0)
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(z, v); 
        z.SetSubVector(fomSp->ess_tdof_list, 0.0);
    }
    z.Neg(); // z = -z, because we are calculating the residual.
    M_hat_solver.Mult(z, dv_dt); // to invert reduced mass matrix operator.

    dx_dt = v;
}



void RomOperator::Mult_FullOrder(const Vector& vx, Vector& dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    Vector x(vx.GetData() + rvdim, rxdim);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);

    // Lift the x-, and v-vectors
    // I.e. perform v = v0 + V_v v^, where v^ is the input
    V_x.mult(x, *z_x);
    V_v.mult(v, *z_v);

    add(z_x, x0, *pfom_x) // Store liftings
    add(z_v, v0, *pfom_v)

    // Apply H to x to get z
    fom->H->Mult(*pfom_x, zfom_x);
    V_x.transposeMult(*zfom_x_librom, z);


    if (fomSp->viscosity != 0.0) 
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(z, v);
        z.SetSubVector(fomSp->ess_tdof_list, 0.0);
    }

    z.Neg(); // z = -z, because we are calculating the residual.
    M_hat_solver.Mult(z, dv_dt); // to invert reduced mass matrix operator.

    dx_dt = v;
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