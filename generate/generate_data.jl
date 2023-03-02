using MultiDimensionalClustering, Random, Enzyme, HDF5, ProgressBars

# random seed for reproducibility
Random.seed!(12345)

# create data directory if it's not there
isdir(pwd() * "/data") ? nothing : mkdir(pwd() * "/data")

include("potential_well.jl")

