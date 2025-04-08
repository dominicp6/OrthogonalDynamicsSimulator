using Statistics
using Molly
using ConstrainedDynamicsSimulator
using Unitful
using CSV
using DataFrames
using .Threads

using Random
Random.seed!(1)

function init_system(logging_interval)
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
            phi=GeneralObservableLogger(phi_wrapper, Float64, logging_interval),
            psi=GeneralObservableLogger(psi_wrapper, Float64, logging_interval),
        ),
        implicit_solvent="gbn2",
    )

    return sys
end

timestep = 0.002u"ps"
fric = 5000.0u"ps^-1"
temps = [30, 300]
traj_length = 100_000
logging_interval = 100
phi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))
psi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))

@threads for idx in 1:length(temps)
    temp = temps[idx]
    sys = init_system(logging_interval)
    temp = temp*u"K"
    simulator = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(dt=timestep, T=temp, γ=fric, 
        φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid, 
        φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat)
    ConstrainedDynamicsSimulator.PVD2!(sys, simulator, traj_length)
    
    # Store data in arrays
    phi_data_array[idx, :] = values(sys.loggers.phi)
    psi_data_array[idx, :] = values(sys.loggers.psi)
    
    # Create time points
    time_points = collect(0:logging_interval:traj_length) * timestep
    
    # Save phi trajectory
    phi_df = DataFrame(
        Time_ps = ustrip.(u"ps", time_points),
        Phi_Angles_deg = phi_data_array[idx, :]
    )
    phi_metadata = """
    # Temperature: $(temp)
    # Timestep: $(timestep)
    # Friction Coefficient: $(fric)
    # Trajectory Length: $(traj_length) steps
    # Logging Interval: $(logging_interval)
    """
    open("./results/trajectories/phi_PVD2_$(temp).csv", "w") do io
        write(io, phi_metadata)
        CSV.write(io, phi_df; append=true, writeheader=true)
    end
    
    # Save psi trajectory
    psi_df = DataFrame(
        Time_ps = ustrip.(u"ps", time_points),
        Psi_Angles_deg = psi_data_array[idx, :]
    )
    psi_metadata = """
    # Temperature: $(temp)
    # Timestep: $(timestep)
    # Friction Coefficient: $(fric)
    # Trajectory Length: $(traj_length) steps
    # Logging Interval: $(logging_interval)
    """
    open("./results/trajectories/psi_PVD2_$(temp).csv", "w") do io
        write(io, psi_metadata)
        CSV.write(io, psi_df; append=true, writeheader=true)
    end
end
