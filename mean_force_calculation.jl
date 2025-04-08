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

# timestep = 0.002u"ps"
final_simulation_time = 0.002  # u"ns"
repeats = 5
timesteps = [0.002 * 10^(x) for x in -3:0.2:0.0]
pushfirst!(timesteps, 0.002 * 10^(-4))
fric = 5000.0u"ps^-1"
temp = 310u"K" #[3.0 * 10^(x) for x in 0.0:0.1:3.0]
logging_interval = 1

timing_results_EM = zeros(repeats, length(timesteps))
mean_force_estimate_EM = zeros(repeats, length(timesteps))

timing_results_PVD2 = zeros(repeats, length(timesteps))
mean_force_estimate_PVD2 = zeros(repeats, length(timesteps))

@threads for repeat_idx in 1:repeats
    for idx in 1:length(timesteps)
        start_time = time_ns()
        timestep = timesteps[idx] * u"ps"
        sys = init_system(logging_interval)
        simulator = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(
            dt=timestep, T=temp, γ=fric,
            φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid,
            φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat
        )
        traj_length = floor(Int, final_simulation_time / ustrip(timestep))
        try
            sys, _, _, _, mean_force_estimate = ConstrainedDynamicsSimulator.euler_maruyama_split_time!(sys, simulator, traj_length, compute_ergodic_integral=true, quantity=ConstrainedDynamicsSimulator.Dihedrals.φ_mean_force)        
            timing_results_EM[repeat_idx, idx] = (time_ns() - start_time) * 1e-9
            mean_force_estimate_EM[repeat_idx, idx] = mean_force_estimate
        catch err
            @warn "EM simulation failed at timestep $timestep: $err"
            continue  # Skip this iteration entirely
        end

        start_time = time_ns()
        timestep = timesteps[idx] * u"ps"
        sys = init_system(logging_interval)
        simulator = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(
            dt=timestep, T=temp, γ=fric,
            φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid,
            φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat
        )
        traj_length = floor(Int, final_simulation_time / ustrip(timestep))
        try
            sys, _, _, _, mean_force_estimate = ConstrainedDynamicsSimulator.PVD2_split_time!(sys, simulator, traj_length, compute_ergodic_integral=true, quantity=ConstrainedDynamicsSimulator.Dihedrals.φ_mean_force)        
            timing_results_PVD2[repeat_idx, idx] = (time_ns() - start_time) * 1e-9
            mean_force_estimate_PVD2[repeat_idx, idx] = mean_force_estimate
        catch err
            @warn "PVD2 simulation failed at timestep $timestep: $err"
            continue  # Skip this iteration entirely
        end
    end
end

# --- Process Results into Tidy DataFrames ---
println("Processing results...")
sys = init_system(1)
sim = ConstrainedDynamicsSimulator.CVConstrainedOverdampedLangevin(
            dt=0.002, T=310, γ=5000,
            φ_grid=ConstrainedDynamicsSimulator.Dihedrals.φ_grid,
            φ_flat=ConstrainedDynamicsSimulator.Dihedrals.φ_flat
        )
phi_value = sim.φ_grid(ustrip.(sys.coords))
results_list = []
for r in 1:repeats
    for i in 1:length(timesteps)
        # Add EM result
        push!(results_list, (
            Repeat = r,
            TimestepIndex = i,
            Timestep_ps = timesteps[i],
            Method = "EM",
            Timing_s = timing_results_EM[r, i],
            MeanForceEstimate = mean_force_estimate_EM[r, i],
            Successful = timing_results_EM[r, i] > 0 # Mark if timing was recorded
        ))
        # Add PVD2 result
        push!(results_list, (
            Repeat = r,
            TimestepIndex = i,
            Timestep_ps = timesteps[i],
            Method = "PVD2",
            Timing_s = timing_results_PVD2[r, i],
            MeanForceEstimate = mean_force_estimate_PVD2[r, i],
            Successful = timing_results_PVD2[r, i] > 0 # Mark if timing was recorded
        ))
    end
end

# Convert list of named tuples to DataFrame
df_results = DataFrame(results_list)

# Separate DataFrames for EM and PVD2 if needed, or keep combined
df_em = filter(row -> row.Method == "EM", df_results)
df_pvd2 = filter(row -> row.Method == "PVD2", df_results)

# --- Save Results ---
println("Saving results...")
output_dir = joinpath(@__DIR__, "results", "final") # Use script directory as base
mkpath(output_dir) # Create directory if it doesn't exist

# Metadata (consistent for both files)
metadata = """
# Julia Molecular Simulation Results
# Temperature: $(temp)
# Phi value: $(phi_value)
# Friction Coefficient: $(fric)
# Target Simulation Time: $(final_simulation_time) ns
# Logging Interval (Steps): $(logging_interval)
# Number of Repeats: $(repeats)
# Timesteps (ps): $(minimum(timesteps)) to $(maximum(timesteps))
# Methods: Euler-Maruyama Split Time (EM), PVD2 Split Time (PVD2)
# System: Alanine Dipeptide (dipeptide_nowater.pdb) with ff99SBildn/TIP3P params, GBN2 implicit solvent
# --- Data Columns ---
# Repeat: Repetition index (1 to $repeats)
# TimestepIndex: Index in the timesteps_ps array
# Timestep_ps: Simulation timestep in picoseconds
# Method: Integration method used (EM or PVD2)
# Timing_s: Wall clock time for the simulation repeat in seconds
# MeanForceEstimate: Mean force estimate value returned by the simulator
# Successful: Boolean indicating if the simulation run completed without error (based on timing > 0)
# ========================
"""

# Save EM Results
output_file_em = joinpath(output_dir, "EM_results_r$(repeats)_t$(final_simulation_time)ns.csv")
try
    open(output_file_em, "w") do io
        write(io, metadata)
        # Add specific method note if needed
        write(io, "# Method: EM\n")
        CSV.write(io, df_em; append=true, writeheader=true) # Append after metadata, write header
    end
    println("EM results saved to: $output_file_em")
catch err
    @error "Failed to save EM results: $err"
end

# Save PVD2 Results
output_file_pvd2 = joinpath(output_dir, "PVD2_results_r$(repeats)_t$(final_simulation_time)ns.csv")
try
    open(output_file_pvd2, "w") do io
        write(io, metadata)
        # Add specific method note if needed
        write(io, "# Method: PVD2\n")
        CSV.write(io, df_pvd2; append=true, writeheader=true) # Append after metadata, write header
    end
    println("PVD2 results saved to: $output_file_pvd2")
catch err
    @error "Failed to save PVD2 results: $err"
end

println("Script finished.")