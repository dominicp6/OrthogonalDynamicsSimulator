module Dihedrals

using LinearAlgebra, StaticArrays

export φ_flat, φ_grid, compute_dihedral


function compute_dihedral(A::AbstractVector, B::AbstractVector, C::AbstractVector, D::AbstractVector)
    # Compute vectors between atoms
    AB = B - A
    BC = C - B
    CD = D - C

    # Cross products
    AB_cross_BC = cross(AB, BC)
    BC_cross_CD = cross(BC, CD)

    # Numerator and denominator for atan2
    numerator = dot(BC, cross(AB_cross_BC, BC_cross_CD))
    denominator = norm(BC) * dot(AB_cross_BC, BC_cross_CD)

    return atan(numerator, denominator)
end

function φ_flat(flattened_coords::AbstractVector)
    # Extract coordinates of the four atoms defining phi
    A = flattened_coords[(5-1)*3 + 1 : 5*3]  # First atom (C')
    B = flattened_coords[(7-1)*3 + 1 : 7*3]  # Second atom (N)
    C = flattened_coords[(9-1)*3 + 1 : 9*3]  # Third atom (Cα)
    D = flattened_coords[(15-1)*3 + 1 : 15*3]  # Fourth atom (C')

    return compute_dihedral(A, B, C, D)
end

function φ_grid(tiled_coords::Vector{SVector{3,Float64}})
    # Extract coordinates of the four atoms defining phi
    A = tiled_coords[5]   # First atom (C')
    B = tiled_coords[7]   # Second atom (N)
    C = tiled_coords[9]   # Third atom (Cα)
    D = tiled_coords[15]  # Fourth atom (C')

    return compute_dihedral(A, B, C, D)
end 

end