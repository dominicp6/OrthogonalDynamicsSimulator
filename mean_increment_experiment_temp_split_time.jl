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

timestep = 0.002u"ps"
fric = 5000.0u"ps^-1"
temps = [3.0 * 10^(x) for x in 0.0:0.1:3.0]
pushfirst!(temps, 3.0)
traj_length = 100
logging_interval = 1
phi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))
psi_data_array = zeros((length(temps), div(traj_length, logging_interval) + 1))
timing_results = zeros(length(temps))
mean_phi_increment = zeros(length(temps))
mean_psi_increment = zeros(length(temps))
mean_phi_curvature = zeros(length(temps))
mean_psi_curvature = zeros(length(temps))
mean_phi_angles = zeros(length(temps))
mean_psi_angles = zeros(length(temps))
std_phi_angles = zeros(length(temps))
std_psi_angles = zeros(length(temps))

@threads for idx in 1:length(temps)
    start_time = time_ns()

    temp = temps[idx]
    sys = init_system(logging_interval)
    temp = temp*u"K"
    simulator = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(dt=timestep, T=temp, γ=fric, φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid, φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat)
    sys, stop_criteria, errors, iterations, _ = ConstrainedDynamicsSimulator.PVD2_split_time_option1!(sys, simulator, traj_length)

    phi_unwrapped = unwrap_angles(values(sys.loggers.phi))
    psi_unwrapped = unwrap_angles(values(sys.loggers.psi))
    
    # First-order differences (D1)
    mean_phi_increment[idx] = mean(abs.(diff(phi_unwrapped)))
    mean_psi_increment[idx] = mean(abs.(diff(psi_unwrapped)))
    
    # Second-order differences (D2) - mean curvature
    phi_second_diff = diff(diff(phi_unwrapped))
    psi_second_diff = diff(diff(psi_unwrapped))
    mean_phi_curvature[idx] = mean(abs.(phi_second_diff))
    mean_psi_curvature[idx] = mean(abs.(psi_second_diff))

    mean_phi_angles = circular_mean(phi_unwrapped)
    mean_psi_angles = circular_mean(psi_unwrapped)
    std_phi_angles = sqrt(circular_variance(phi_unwrapped))
    std_psi_angles = sqrt(circular_variance(psi_unwrapped))

    # Step 1: Create the iteration count statistics
    unique_iterations = unique(iterations)  # Get unique iteration counts (e.g., 1, 2, 3, etc.)
    iteration_counts = Dict()  # To store the count of each iteration number
    for iter in unique_iterations
        iteration_counts[iter] = count(x -> x == iter, iterations)  # Count occurrences of each iteration number
    end

    # Step 2: Calculate the mean error for each iteration count
    mean_errors = Dict()  # To store the mean error for each iteration number
    for iter in unique_iterations
        # Find the indices where the iteration count matches `iter`
        indices = findall(x -> x == iter, iterations)
        # Extract the corresponding errors
        iter_errors = errors[indices]
        # Calculate the mean of the errors for this iteration count
        mean_errors[iter] = mean(iter_errors)
    end

    # Step 3: Prepare data for CSV output
    # Prepare a DataFrame for easy CSV export
    results_df = DataFrame(
        "Iteration_Count" => unique_iterations,
        "Num_Steps" => [iteration_counts[iter] for iter in unique_iterations],
        "Mean_Error" => [mean_errors[iter] for iter in unique_iterations]
    )

    # Step 4: Write the DataFrame to a CSV file
    CSV.write("iteration_$idx.csv", results_df)

    timing_results[idx] = (time_ns() - start_time) * 1e-9
end

df = DataFrame(
    Temperature_K = temps,
    Mean_Phi_Angles = mean_phi_angles,
    Std_Phi_Angles = std_phi_angles,
    Mean_Phi_Increment = mean_phi_increment,
    Mean_Phi_Curvature = mean_phi_curvature,  # Added D2 metric
    Mean_Psi_Angles = mean_psi_angles,
    Std_Psi_Angles = std_psi_angles,
    Mean_Psi_Increment = mean_psi_increment,
    Mean_Psi_Curvature = mean_psi_curvature,  # Added D2 metric
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

open("./results/final/mean_and_std_angles_PVD2_x_$(traj_length).csv", "w") do io
    write(io, metadata)
    CSV.write(io, df; append=true, writeheader=true)
end

println("Results saved to mean_and_std_angles.csv")