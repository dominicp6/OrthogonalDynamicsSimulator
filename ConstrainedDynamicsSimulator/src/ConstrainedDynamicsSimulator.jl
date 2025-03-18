module ConstrainedDynamicsSimulator

# Load dependencies
using Molly
using Zygote
using ForwardDiff
using LinearAlgebra
using StaticArrays

# Include separate files for modularity 
include("calculus.jl")
include("SDE_noise_integrators.jl")
include("molly_integrators.jl")
include("dihedrals.jl")


end # module ConstrainedDynamicsSimulator
