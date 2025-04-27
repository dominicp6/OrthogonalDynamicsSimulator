using Statistics
using Molly
using ConstrainedDynamicsSimulator
using Unitful
using CSV
using DataFrames
using .Threads
using Base: time_ns

# Define the unwrapping function
function unwrap_angles(angles)
    unwrapped = similar(angles)
    unwrapped[1] = angles[1]
    for i in 2:length(angles)
        Δ = angles[i] - angles[i-1]
        Δ_adj = mod(Δ + 180, 360) - 180
        unwrapped[i] = unwrapped[i-1] + Δ_adj
    end
    return unwrapped
end

# Function to compute circular mean of angles
function circular_mean(angles)
    rad_angles = deg2rad.(angles)
    sum_sin = sum(sin.(rad_angles))
    sum_cos = sum(cos.(rad_angles))
    mean_rad = atan(sum_sin, sum_cos)
    mean_deg = rad2deg(mean_rad)
    return mean_deg
end

# Function to compute circular variance of angles
function circular_variance(angles)
    rad_angles = deg2rad.(angles)
    sum_sin = sum(sin.(rad_angles))
    sum_cos = sum(cos.(rad_angles))
    R = sqrt(sum_sin^2 + sum_cos^2) / length(angles)
    var = 1 - R
    return var
end

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

# timestep = 0.002u"ps"
timesteps = [0.002 * 10^(x) for x in -2:0.1:1.0]
fric = 5000.0u"ps^-1"
temp = 310u"K" #[3.0 * 10^(x) for x in 0.0:0.1:3.0]
pushfirst!(timesteps, 0.00002)
traj_length = 100
logging_interval = 1
timing_results = zeros(length(timesteps))
mean_phi_increment = zeros(length(timesteps))
mean_psi_increment = zeros(length(timesteps))
mean_phi_curvature = zeros(length(timesteps))
mean_psi_curvature = zeros(length(timesteps))
mean_phi_angles = zeros(length(timesteps))
mean_psi_angles = zeros(length(timesteps))
std_phi_angles = zeros(length(timesteps))
std_psi_angles = zeros(length(timesteps))

@threads for idx in 1:length(timesteps)
    start_time = time_ns()
    timestep = timesteps[idx] * u"ps"
    sys = init_system(logging_interval)
    simulator = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(
        dt=timestep, T=temp, γ=fric,
        φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid,
        φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat
    )

    try
        sys, _ = ConstrainedDynamicsSimulator.PVD2!(sys, simulator, traj_length)
        phi_unwrapped = unwrap_angles(values(sys.loggers.phi))
        psi_unwrapped = unwrap_angles(values(sys.loggers.psi))
        
        # Store observable data
        mean_phi_angles[idx] = circular_mean(phi_unwrapped)
        mean_psi_angles[idx] = circular_mean(psi_unwrapped)
        std_phi_angles[idx] = sqrt(circular_variance(phi_unwrapped))
        std_psi_angles[idx] = sqrt(circular_variance(psi_unwrapped))
        mean_phi_increment[idx] = mean(abs.(diff(phi_unwrapped)))
        mean_psi_increment[idx] = mean(abs.(diff(psi_unwrapped)))
        mean_phi_curvature[idx] = mean(abs.(diff(diff(phi_unwrapped))))
        mean_psi_curvature[idx] = mean(abs.(diff(diff(psi_unwrapped))))
        
        timing_results[idx] = (time_ns() - start_time) * 1e-9
    catch err
        @warn "Simulation failed at timestep $timestep: $err"
        continue  # Skip this iteration entirely
    end
end

# Identify successful runs (nonzero timing results)
valid_indices = findall(x -> x > 0, timing_results)

df = DataFrame(
    Timesteps_ps = timesteps[valid_indices],
    Mean_Phi_Angles = mean_phi_angles[valid_indices],
    Std_Phi_Angles = std_phi_angles[valid_indices],
    Mean_Phi_Increment = mean_phi_increment[valid_indices],
    Mean_Phi_Curvature = mean_phi_curvature[valid_indices],
    Mean_Psi_Angles = mean_psi_angles[valid_indices],
    Std_Psi_Angles = std_psi_angles[valid_indices],
    Mean_Psi_Increment = mean_psi_increment[valid_indices],
    Mean_Psi_Curvature = mean_psi_curvature[valid_indices],
    Timing_Results = timing_results[valid_indices]
)

metadata = """
# Temperature: $(temp)
# Friction Coefficient: $(fric)
# Trajectory Length: $(traj_length) steps
# Logging Interval: $(logging_interval)
# Stepsize Range: $(minimum(timesteps)) ps to $(maximum(timesteps)) ps
# Number of Timesteps: $(length(timesteps))
"""

open("./results/final/stepsize_scaling_PVD2_x_bar_$(traj_length).csv", "w") do io
    write(io, metadata)
    CSV.write(io, df; append=true, writeheader=true)
end

println("Results saved to mean_and_std_angles.csv")