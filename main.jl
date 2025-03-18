using Revise
using Molly
using Unitful
using ConstrainedDynamicsSimulator

ff_dir = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields")
ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
sys = System("dipeptide_nowater.pdb", ff; rename_terminal_res=false)

function phi_wrapper(sys, args...; kwargs...)
    rad2deg(torsion_angle(sys.coords[5], sys.coords[7], sys.coords[9],
                          sys.coords[15], sys.boundary))
end

function psi_wrapper(sys, args...; kwargs...)
    rad2deg(torsion_angle(sys.coords[7], sys.coords[9], sys.coords[15],
                          sys.coords[17], sys.boundary))
end

sys = System(
    "dipeptide_nowater.pdb",
    ff;
    rename_terminal_res=false,
    loggers=(
        writer=StructureWriter(100, "trajectory.pdb"),
        phi=GeneralObservableLogger(phi_wrapper, Float64, 100),
        psi=GeneralObservableLogger(psi_wrapper, Float64, 100),
    ),
    implicit_solvent="gbn2",
)

temp = 300.0u"K"
timestep = 0.002u"ps"
fric = 5000.0u"ps^-1"
simulator = CVConstrainedOverdampedLangevin(dt=timestep, T=temp, γ=fric, φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid, φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat)

ConstrainedDynamicsSimulator.simulate!(sys, simulator, 2_000) # This will take a little while to run


