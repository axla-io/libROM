// First steps on the path towards CWROM implementation.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char* argv[])
{

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    const char* mesh_file1 = "../data/rhino1.mesh";
    const char* mesh_file2 = "../data/rhino2.mesh";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file1, "-m1", "--mesh1",
        "First mesh file to use.");
    args.AddOption(&mesh_file2, "-m2", "--mesh2",
        "Second mesh file to use.");

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

    // 3. Device configuration


    // 4. Read the (serial) meshes for the components.
    Mesh mesh1(mesh_file1, 1, 1);
    Mesh mesh2(mesh_file2, 1, 1);
    int dim = mesh1.Dimension();


    // 5. Refine mesh


    // 6. Define parallel meshes by a partitioning of the serial meshes.
    ParMesh* pmesh1(MPI_COMM_WORLD, mesh1);
    ParMesh* pmesh2(MPI_COMM_WORLD, mesh2);

    mesh1.Clear();
    mesh2.Clear();


    // 8. Solve the problem approximately using 
    // FEM once on the reference domain to generate 
    // bubble function approximation.
    
    // 8a. Define a finite element space on the meshes
    FiniteElementCollection* fec1;
    ParFiniteElementSpace* fespace1;
    const bool use_nodal_fespace = pmesh1->NURBSext && !amg_elast;
    if (use_nodal_fespace)
    {
        cout << "Test1";
        fec1 = NULL;
        fespace1 = (ParFiniteElementSpace*)pmesh1->GetNodes()->FESpace();
    }
    else
    {
        cout << "Test2";
        fec1 = new H1_FECollection(order, dim);
        if (reorder_space)
        {
            cout << "Test3";
            fespace1 = new ParFiniteElementSpace(pmesh1, fec1, dim, Ordering::byNODES);
        }
        else
        {
            cout << "Test4";
            fespace1 = new ParFiniteElementSpace(pmesh1, fec1, dim, Ordering::byVDIM);
        }
    }


    HYPRE_BigInt size = fespace1->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl
            << "Assembling: " << flush;
    }


    

    // 9. Define linear and bilinear forms

    // 10. Assembly

    // 11. Training with governing equation




    cout << "All good!";

	return 1;
}