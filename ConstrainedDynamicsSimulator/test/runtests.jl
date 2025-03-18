using Test
using StaticArrays
using LinearAlgebra
using Zygote
using Molly
include("../src/molly_integrators.jl") 
include("../src/calculus.jl")
include("../src/dihedrals.jl")
using .Dihedrals

# Define configurations
const grid_config = [
    SVector(16.548, 15.057, 17.037),
    SVector(16.003, 15.960, 16.763),
    SVector(15.137, 15.987, 17.425),
    SVector(16.000, 16.042, 15.676),
    SVector(16.745, 17.229, 17.149),
    SVector(17.845, 17.104, 17.750),
    SVector(16.324, 18.333, 16.582),
    SVector(15.385, 18.357, 16.213),
    SVector(17.074, 19.555, 16.665),
    SVector(17.250, 19.742, 17.725),
    SVector(18.392, 19.541, 15.893),
    SVector(18.071, 19.476, 14.854),
    SVector(18.926, 20.477, 16.059),
    SVector(19.041, 18.750, 16.270),
    SVector(16.135, 20.774, 16.356),
    SVector(15.200, 20.621, 15.614),
    SVector(16.567, 22.016, 16.783),
    SVector(17.473, 22.130, 17.214),
    SVector(15.941, 23.320, 16.599),
    SVector(14.862, 23.237, 16.466),
    SVector(16.109, 23.973, 17.455),
    SVector(16.354, 23.845, 15.738)
]

const flat_config = [
    16.548, 15.057, 17.037,
    16.003, 15.960, 16.763,
    15.137, 15.987, 17.425,
    16.000, 16.042, 15.676,
    16.745, 17.229, 17.149,
    17.845, 17.104, 17.750,
    16.324, 18.333, 16.582,
    15.385, 18.357, 16.213,
    17.074, 19.555, 16.665,
    17.250, 19.742, 17.725,
    18.392, 19.541, 15.893,
    18.071, 19.476, 14.854,
    18.926, 20.477, 16.059,
    19.041, 18.750, 16.270,
    16.135, 20.774, 16.356,
    15.200, 20.621, 15.614,
    16.567, 22.016, 16.783,
    17.473, 22.130, 17.214,
    15.941, 23.320, 16.599,
    14.862, 23.237, 16.466,
    16.109, 23.973, 17.455,
    16.354, 23.845, 15.738
]

# Define system
ff_dir = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields")
ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
sys = System("../../dipeptide_nowater.pdb", ff; rename_terminal_res=false)

# DIHEDRALS TESTS
# TODO: Add checks against Molly's computation of this angle
@testset "Coordinates and dihedrals" begin
    @test flatten(grid_config) ≈ flat_config
    @test unflatten(flat_config) ≈ grid_config 

    # Test φ_flat correctness
    @test φ_flat(flat_config) ≈ compute_dihedral(
        [16.745, 17.229, 17.149],  # C atom from ACE (residue 1)
        [16.324, 18.333, 16.582],  # N atom of ALA (residue 2)
        [17.074, 19.555, 16.665],  # CA atom of ALA (residue 2)
        [16.135, 20.774, 16.356]   # C atom of ALA (residue 2)
    )

    # Test φ_grid correctness
    @test φ_grid(grid_config) ≈ compute_dihedral(
        [16.745, 17.229, 17.149],  # C atom from ACE (residue 1)
        [16.324, 18.333, 16.582],  # N atom of ALA (residue 2)
        [17.074, 19.555, 16.665],  # CA atom of ALA (residue 2)
        [16.135, 20.774, 16.356]   # C atom of ALA (residue 2)
    )

    # Test consistency between φ_flat and φ_grid
    @test φ_flat(flat_config) ≈ φ_grid(grid_config)

    # Test consistency with Molly computation
    torsion_angle(grid_config[5]*u"nm", grid_config[7]*u"nm", grid_config[9]*u"nm", grid_config[15]*u"nm", sys.boundary) ≈ compute_dihedral(
        [16.745, 17.229, 17.149],  # C atom from ACE (residue 1)
        [16.324, 18.333, 16.582],  # N atom of ALA (residue 2)
        [17.074, 19.555, 16.665],  # CA atom of ALA (residue 2)
        [16.135, 20.774, 16.356]   # C atom of ALA (residue 2)
    )
end

# CALCULUS TESTS
@testset "Clean gradient" begin
    # Test clean_gradient
    @test clean_gradient([nothing, SVector{3, Float64}(1.0, 2.0, 3.0), nothing]) == [
        SVector{3, Float64}(0.0, 0.0, 0.0),
        SVector{3, Float64}(1.0, 2.0, 3.0),
        SVector{3, Float64}(0.0, 0.0, 0.0)
    ] 
end

# Test gradient of φ_grid
@testset "Gradient of φ_grid" begin
    # Compute the gradient using Zygote
    grad_phi_grid = clean_gradient(Zygote.gradient(φ_grid, grid_config)[1])
    
    # Extract coordinates
    x5 = grid_config[5]   # A, C(i-1)
    x7 = grid_config[7]   # B, N(i)
    x9 = grid_config[9]   # C, Cα(i)
    x15 = grid_config[15] # D, C(i)

    # Define vectors
    a = x7 - x5
    b_vec = x9 - x7
    c = x15 - x9
    b = norm(b_vec)
    s = dot(a, cross(b_vec, c))
    p = dot(a, b_vec)
    q = dot(b_vec, c)
    r = dot(a, c)

    # Compute expected gradient
    numerator = -b * cross(b_vec, c) * (p * q - r * b^2) + s * b * (q * b_vec - b^2 * c)
    denominator = (p * q - r * b^2)^2 + (s * b)^2
    expected_grad = numerator / denominator

    # Test
    @test grad_phi_grid[5] ≈ expected_grad
end

# Test gradient of φ_flat
@testset "Gradient of φ_flat" begin
    # Compute the gradient using Zygote
    grad_phi_flat = clean_gradient(Zygote.gradient(φ_flat, flat_config)[1])
    
    # Extract coordinates
    x5 = flat_config[(5-1)*3 + 1 : 5*3]
    x7 = flat_config[(7-1)*3 + 1 : 7*3]
    x9 = flat_config[(9-1)*3 + 1 : 9*3]
    x15 = flat_config[(15-1)*3 + 1 : 15*3]

    # Define vectors
    a = x7 - x5
    b_vec = x9 - x7
    c = x15 - x9
    b = norm(b_vec)
    s = dot(a, cross(b_vec, c))
    p = dot(a, b_vec)
    q = dot(b_vec, c)
    r = dot(a, c)

    # Compute expected gradient
    numerator = -b * cross(b_vec, c) * (p * q - r * b^2) + s * b * (q * b_vec - b^2 * c)
    denominator = (p * q - r * b^2)^2 + (s * b)^2
    expected_grad = numerator / denominator

    # Test
    @test grad_phi_flat[(5-1)*3 + 1 : 5*3] ≈ expected_grad
end

@testset "Flatten and unflatten" begin
    test_grid = [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
    test_flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test flatten(test_grid) ≈ test_flat
    @test unflatten(test_flat) ≈ test_grid
end

# @testset "Random noise with SVector shape" begin
    
# end

# Test compute_Pvec variants
@testset "Projection computations" begin

    @testset "Tests of SVector linear algebra" begin
        test_grad = [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
        @test dot(test_grad, test_grad) ≈ 91.0
        test_vec = [SVector(0.2, 0.1, 0.6), SVector(-0.2, -0.1, 0.3)]
        @test dot(test_grad, test_vec) ≈ 2.7
        @test test_vec .- test_grad .* (dot(test_grad, test_vec)) ./ dot(test_grad, test_grad) ≈ [SVector(0.17032967032967034, 0.04065934065934068, 0.510989010989011), SVector(-0.31868131868131866, -0.24835164835164833, 0.12197802197802202)]
    end

    # General test for compute_Pvec
    @testset "General compute_Pvec" begin
        grad_phi = [1.0, -2.0, 1.5, -0.5, 2.0, -1.0]
        v = [2.0, -0.5, 1.0, 6.0, 3.0, -2.0]
        P_explicit = I(6) .- (grad_phi * grad_phi') ./ dot(grad_phi, grad_phi)
        hand_crafted_P = [1-(1.0)^2/12.5 -(1.0)*(-2.0)/12.5 -(1.0)*1.5/12.5 -(1.0)*(-0.5)/12.5 -(1.0)*2.0/12.5 -(1.0)*(-1.0)/12.5; -(-2.0)*1.0/12.5 1-(-2.0)^2/12.5 -(-2.0)*1.5/12.5 -(-2.0)*(-0.5)/12.5 -(-2.0)*2.0/12.5 -(-2.0)*(-1.0)/12.5; -1.5*1.0/12.5 -1.5*(-2.0)/12.5 1-(1.5)^2/12.5 -1.5*(-0.5)/12.5 -1.5*2.0/12.5 -1.5*(-1.0)/12.5; -(-0.5)*1.0/12.5 -(-0.5)*(-2.0)/12.5 -(-0.5)*1.5/12.5 1-(-0.5)^2/12.5 -(-0.5)*2.0/12.5 -(-0.5)*(-1.0)/12.5; -2.0*1.0/12.5 -2.0*(-2.0)/12.5 -2.0*1.5/12.5 -2.0*(-0.5)/12.5 1-(2.0)^2/12.5 -2.0*(-1.0)/12.5; -(-1.0)*1.0/12.5 -(-1.0)*(-2.0)/12.5 -(-1.0)*1.5/12.5 -(-1.0)*(-0.5)/12.5 -(-1.0)*2.0/12.5 1-(-1.0)^2/12.5]
        
        grad_phi_grid = [SVector(1.0, -2.0, 1.5), SVector(-0.5, 2.0, -1.0)]
        v_grid = [SVector(2.0, -0.5, 1.0), SVector(6.0, 3.0, -2.0)]
        @test P_explicit * v ≈ flatten(compute_Pvec(gradφ=grad_phi_grid, vec=v_grid))
        @test P_explicit ≈ hand_crafted_P
        @test hand_crafted_P * v ≈ flatten(compute_Pvec(gradφ=grad_phi_grid, vec=v_grid))
    end

    # Test compute_Pvec with grid_config
    @testset "compute_Pvec with grid_config" begin
        grad_phi = clean_gradient(Zygote.gradient(φ_grid, grid_config)[1])
        grad_phi_flat = flatten(grad_phi)
        v = sample_noise_vector(length(grad_phi))
        v_flat = flatten(v)
        P_explicit = I(length(grad_phi_flat)) - (grad_phi_flat * grad_phi_flat') ./ dot(grad_phi_flat, grad_phi_flat)
        @test P_explicit * v_flat ≈ flatten(compute_Pvec(gradφ=grad_phi, vec=v))
    end

    # Test compute_Pvec_x with grid_config
    @testset "compute_Pvec_x with grid_config" begin
        grad_phi = clean_gradient(Zygote.gradient(φ_grid, grid_config)[1])
        grad_phi_flat = flatten(grad_phi)
        v = sample_noise_vector(length(grad_phi))
        v_flat = flatten(v)
        P_explicit = I(length(grad_phi_flat)) - (grad_phi_flat * grad_phi_flat') ./ dot(grad_phi_flat, grad_phi_flat)
        @test P_explicit * v_flat ≈ flatten(compute_Pvec_x(φ_grid=φ_grid, vec=v, x=grid_config)[1])
    end

    # Test compute_Pvec! with grid_config
    @testset "compute_Pvec! with grid_config" begin
        grad_phi = clean_gradient(Zygote.gradient(φ_grid, grid_config)[1])
        grad_phi_flat = flatten(grad_phi)
        v = sample_noise_vector(length(grad_phi))
        v_flat = flatten(v)
        Pvec = similar(grad_phi)
        P_explicit = I(length(grad_phi_flat)) - (grad_phi_flat * grad_phi_flat') ./ dot(grad_phi_flat, grad_phi_flat)
        compute_Pvec!(Pvec, gradφ=grad_phi, vec=v)
        @test P_explicit * v_flat ≈ flatten(Pvec)
    end

    # Test compute_Pvec_x! with grid_config
    @testset "compute_Pvec_x! with grid_config" begin
        grad_phi = clean_gradient(Zygote.gradient(φ_grid, grid_config)[1])
        grad_phi_flat = flatten(grad_phi)
        v = sample_noise_vector(length(grad_phi))
        v_flat = flatten(v)
        Pvec = similar(grad_phi)
        P_explicit = I(length(grad_phi_flat)) - (grad_phi_flat * grad_phi_flat') ./ dot(grad_phi_flat, grad_phi_flat)
        compute_Pvec_x!(Pvec, grad_phi, φ_grid=φ_grid, vec=v, x=grid_config)
        @test P_explicit * v_flat ≈ flatten(Pvec)
    end
end

@testset "Divergence tests" begin
    g = [1, 2, 3]
    H = [7 3 6; -2 -5 4; 1 8 -9]
    laplacian = sum(diag(H))        # ∇²φ = tr(H)
    Hg = H * g                      # H ∇φ
    gHg = dot(g, Hg)                # ∇φ^T H ∇φ
    norm_g_sq = dot(g, g)           # ||∇φ||²
    div_P = - (laplacian / norm_g_sq) * g + (2 * gHg / (norm_g_sq^2)) * g - (1 / norm_g_sq) * Hg    
    @test div_P ≈ [-167/98, 50/49, 110/49]
end