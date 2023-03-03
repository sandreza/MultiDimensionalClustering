using MultiDimensionalClustering, Random, Enzyme, HDF5, ProgressBars

# random seed for reproducibility
Random.seed!(12345)

# create data directory if it's not there
isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")

# potential well
if isfile(pwd() * "/data/potential_well.hdf5")
    @info "potential well data already exists. skipping data generation"
else 
    include("potential_well.jl")
end

# lorenz
if isfile(pwd() * "/data/lorenz.hdf5")
    @info "lorenz data already exists. skipping data generation"
else
    include("lorenz.jl")
end

