
void RomOperator::Mult_Hyperreduced(const Vector& vx, Vector& dvx_dt) const
{
    // Assuming that the input is vx in generalized coordinates...
    // Calculate V_vx^T M^-1 (H_red(x0 + V_vx x^) + S (v0 + V_vx v^)
    // Where V_vx is the (reduced) basis for the solution space.
    // And H_red is the reduced non linear operator 
    // H_red = V_H H^
    // H^ = (Z^T V_H)^-1 * Z^T * H_sample
    // V_H is the (reduced) basis for H


    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rrdim && dvx_dt.Size() == rrdim, ""); // rrdim should be renamed


    // Lift the input vectors
    // I.e. perform vx = vx0 + V_vx vx^, where vx^ is the input
    V_vx.mult(vx, *zfom_vx_librom);
    add(zfom_vx_librom, vx0, fom_vx_librom) // Get initial conditions stored in class, also fom_vx


    // Create temporary vectors
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
        int sc = height / 2;
    CAROM::Vector v_librom(fom_vx_librom.GetData() + 0, sc);
    CAROM::Vector x_librom(fom_vx_librom.GetData() + sc, sc);
    CAROM::Vector dv_dt_librom(dvx_dt.GetData() + 0, sc);
    CAROM::Vector dx_dt_librom(dvx_dt.GetData() + sc, sc);


    // Hyper-reduction
    smm->GetSampledValues("H", zR, zN); // Sample H according to zR, to generate zN

    // Get H^ in general coordinates
    if (oversampling)
    {
        Hsinv->transposeMult(zN, zH); // This is H^
    }
    else
    {
        Hsinv->mult(zN, zH); // This is H^
    }


    // Apply H_basis
    V_H.Mult(zH, H_red);

    // Apply approximated operator
    H_red.Mult(x_librom, z_librom);


    // The Laplacian operator is linear so we don't hyperreduce it...
    if (viscosity != 0.0)
    {
        fom->S.TrueAddMult(v_librom, z_librom);
        z_librom.SetSubVector(ess_tdof_list, 0.0);
    }

    z_librom.Neg(); // z = -z

    // Also kept the same, since the inverted mass matrix doesn't change
    fom->M_solver.Mult(z_librom, dv_dt_temp);
    V_vx.transposeMult(dv_dt_temp, dv_dt_librom);


    dx_dt_librom = v_librom;
    dvxdt_prev = dvx_dt;

}



void RomOperator::Mult_FullOrder(const Vector& vx, Vector& dvx_dt) const
{
    // Assuming that the input is vx in generalized coordinates...
    // Calculate V_vx^T M^-1 (H(x0 + V_vx x^) + S (v0 + V_vx v^)
    // Where V_vx is the (reduced) basis for the solution space.

    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rrdim && dvx_dt.Size() == rrdim, ""); // rrdim should be renamed


    // Lift the input vectors
    // I.e. perform vx = vx0 + V_vx vx^, where vx^ is the input
    V_vx.mult(vx, *zfom_vx_librom);
    add(zfom_vx_librom, vx0, fom_vx_librom) // Get initial conditions stored in class, also fom_vx

    // Create temporary vectors
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
        int sc = height / 2;
    CAROM::Vector v_librom(fom_vx_librom.GetData() + 0, sc);
    CAROM::Vector x_librom(fom_vx_librom.GetData() + sc, sc);
    CAROM::Vector dv_dt_librom(dvx_dt.GetData() + 0, sc);
    CAROM::Vector dx_dt_librom(dvx_dt.GetData() + sc, sc);

    fom->H.Mult(x_librom, z_librom);
    if (viscosity != 0.0)
    {
        fom->S.TrueAddMult(v_librom, z_librom);
        z_librom.SetSubVector(ess_tdof_list, 0.0);
    }
    z_librom.Neg(); // z = -z
    fom->M_solver.Mult(z_librom, dv_dt_temp);
    V_R.transposeMult(dv_dt_temp, dv_dt_librom);


    dx_dt_librom = v_librom;
    dvxdt_prev = dvx_dt;
}



// Hyperelastic operator for reference
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
    dvxdt_prev = dvx_dt;
}