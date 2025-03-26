module ConstrainedDynamicsSimulator

# Load dependencies
using Molly
using Zygote
using ForwardDiff
using LinearAlgebra
using StaticArrays

struct CVConstrainedOverdampedLangevin{picosecond, kelvin, inverse_picosecond, F1, F2} 
    dt::picosecond
    T::kelvin
    γ::inverse_picosecond
    φ_grid::F1
    φ_flat::F2
    remove_CM_motion::Int
end

CVConstrainedOverdampedLangevin(; dt, T, γ, φ_grid, φ_flat, remove_CM_motion=1) = CVConstrainedOverdampedLangevin(dt, T, γ, φ_grid, φ_flat, Int(remove_CM_motion))

# Include separate files for modularity 
include("calculus.jl")
include("noise_integrators.jl")
include("integrators.jl")
include("dihedrals.jl")


end # module ConstrainedDynamicsSimulator
