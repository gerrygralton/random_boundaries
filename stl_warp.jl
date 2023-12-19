using gmsh
using LinearAlgebra
using GaussianRandomFields

# Example covariance structures
cov_struct = Linear(150., σ=0.627/4)
# cov_struct = SquaredExponential(50., σ=0.627/4)
# cov_struct = Matern(50., 2.5, σ=0.627/4)

# file - path to file
# num_samples - number of random geometries to generate
# cov_struct - covaraince structure. see examples above
function create_random_models(file::String, num_samples::Int64, cov_struct::CovarianceStructure)
    gmsh.initialize()
    gmsh.clear()

    gmsh.merge(file)

    gmsh.model.mesh.classifySurfaces(2, true, true)
    gmsh.model.mesh.createGeometry()

    # Make sure that our surface mesh has enough
    # resolution to describe both the covariance structure and the actual geometry
    bound_box = gmsh.model.getBoundingBox(-1, -1)
    dims = length(bound_box)/2
    min_len = minimum( abs.(bound_box[1:dims] .- bound_box[dims+1:end]))
    cov_struct.λ < min_len ? len_scale = cov_struct.λ * 0.05 : len_scale = min_len * 0.05

    gmsh.model.mesh.set_size(gmsh.model.getEntities(0), len_scale)
    gmsh.model.mesh.generate(2)

    entities = gmsh.model.getEntities(2)

    normals = Vector{Float64}(undef, 0)
    coords = Vector{Float64}(undef, 0)
    tags = UInt64[]

    for e in entities
        n_tags, n_coords, n_params = gmsh.model.mesh.getNodes(e[1], e[2], true, true)

        tags = vcat(tags, n_tags)

        coords = vcat(coords, n_coords)
        normals = vcat(normals, gmsh.model.getNormal(e[2], n_params))
    end

    coords = permutedims(reshape(coords, (3, length(tags))))
    normals = permutedims(reshape(normals, (3, length(tags))))

    normals = normals ./ norm.(eachrow(normals))

    cov_fun = CovarianceFunction(3, cov_struct)
    grf = GaussianRandomField(cov_fun, KarhunenLoeve(1000), coords)

    for i in 1:num_samples
        # Make a random draw from our mulitvariate normal distribution.
        # This draw is the scale of the pertubation at each node
        noise = sample(grf)

        # Add the randomly drawn pertubation at each node to the node's position
        # in the direction normal to the surface.
        additive = normals .* noise
        new_coords = coords .+ additive

        for i in 1:length(tags)
            gmsh.model.mesh.setNode(tags[i], new_coords[i, :], [])
        end

        # Uncomment the next line to visualise your mesh
        # gmsh.fltk.run()

        # Export the perturbed model
        gmsh.write(string("random_", i, ".stl"))

        for i in 1:length(tags)
            gmsh.model.mesh.setNode(tags[i], -new_coords[i, :], [])
        end
    end
    gmsh.clear()
end
