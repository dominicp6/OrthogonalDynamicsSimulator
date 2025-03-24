using Statistics
using Molly
using ConstrainedDynamicsSimulator
using Unitful
using CSV
using DataFrames
using .Threads
using Base: time_ns

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
temps = [3.0 * 10^(x) for x in 0:0.1:3]
pushfirst!(temps, 3.0)
traj_length = 100_000
logging_interval = 1
phi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))
psi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))
# stopping_condition_data = zeros((length(temps), 3))
timing_results = zeros(length(temps))
mean_phi_increment = zeros(length(temps))
mean_psi_increment = zeros(length(temps))

for idx in 1:length(temps)
    start_time = time_ns()

    temp = temps[idx]
    sys = init_system(logging_interval)
    temp = temp*u"K"
    simulator = CVConstrainedOverdampedLangevin(dt=timestep, T=temp, γ=fric, φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid, φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat)
    sys = ConstrainedDynamicsSimulator.simulate!(sys, simulator, traj_length)
    phi_data_array[idx, :] = values(sys.loggers.phi)
    psi_data_array[idx, :] = values(sys.loggers.psi)
    mean_phi_increment[idx] = mean(abs.(diff(values(sys.loggers.phi))))
    mean_psi_increment[idx] = mean(abs.(diff(values(sys.loggers.psi))))
    # stopping_condition_data[idx, :] = stopping_condition_counts

    timing_results[idx] = (time_ns() - start_time) * 1e-9
end

mean_phi_angles = vec(mean(phi_data_array, dims=2))
mean_psi_angles = vec(mean(psi_data_array, dims=2))
std_phi_angles = vec(std(phi_data_array, dims=2))
std_psi_angles = vec(std(psi_data_array, dims=2))

df = DataFrame(
    Temperature_K = temps,
    Mean_Phi_Angles = mean_phi_angles,
    Std_Phi_Angles = std_phi_angles,
    Mean_Phi_Increment = mean_phi_increment,
    Mean_Psi_Angles = mean_psi_angles,
    Std_Psi_Angles = std_psi_angles,
    Mean_Psi_Increment = mean_psi_increment,
    # Stopping1 = stopping_condition_data[:, 1],
    # Stopping2 = stopping_condition_data[:, 2],
    # Stopping3 = stopping_condition_data[:, 3],
    Timing_Results = timing_results
)

metadata = """
# Timestep: $(timestep)
# Friction Coefficient: $(fric)
# Trajectory Length: $(traj_length) steps
# Logging Interval: $(logging_interval)
# Temperature Range: $(minimum(temps)) K to $(maximum(temps)) K
# Number of Temperatures: $(length(temps))
"""

open("./results/EM/mean_and_std_angles_with_times_$(traj_length).csv", "w") do io
    write(io, metadata)
    CSV.write(io, df; append=true, writeheader=true)
end

println("Results saved to mean_and_std_angles.csv")