using Molly
using Unitful
using ConstrainedDynamicsSimulator

ff_dir = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields")
ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
sys = System("dipeptide_nowater.pdb", ff; rename_terminal_res=false)

function phi_wrapper(sys, args...; kwargs...)
    rad2deg(torsion_angle(sys.coords[2], sys.coords[7], sys.coords[9],
                          sys.coords[15], sys.boundary))
end

function psi_wrapper(sys, args...; kwargs...)
    rad2deg(torsion_angle(sys.coords[7], sys.coords[9], sys.coords[15],
                          sys.coords[17], sys.boundary))
end

const boundary = sys.boundary 
function φ(x)
    return torsion_angle(x[7], x[9], x[15], x[17], boundary)
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

T = 300.0u"K"
kB = 1.380649e-23u"J/K"  # Boltzmann constant in joules per kelvin
simulator = ConstrainedIntegrator_PVD2(dt=0.0005u"ps", sigma=sqrt(2 * kB * T), φ=φ)


ConstrainedDynamicsSimulator.simulate!(sys, simulator, 200_000) # This will take a little while to run
