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
    const char* mesh_file = "../data/rhino.mesh";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
        "Mesh file to use.");

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


    // 4. Read the (serial) mesh for the reference component.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();


    // 5. Refine mesh


    // 6. Define a parallel mesh by a partitioning of the serial mesh.
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();



    // 7. Define finite element space on mesh



    // 8. Solve the problem approximately using 
    // FEM once on the reference domain to generate 
    // bubble function approximation.

    // 9. Define linear and bilinear forms

    // 10. Assembly

    // 11. Training with governing equation




    cout << "All good!";

	return 1;
}