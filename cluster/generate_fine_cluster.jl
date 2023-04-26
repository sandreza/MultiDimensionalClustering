using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
Random.seed!(12345)
clusters_fine = 1000
files = ["potential_well", "lorenz", "newton"]
check_file = true
for (i, file) in ProgressBar(enumerate(files))
    @info "opening data for $file"
    hfile = h5open("data/" * file * ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    if isfile(pwd() * "/data/" * file * "_fine_cluster.hdf5") && check_file
        @info "data already exists. skipping data generation"
    else
        @info "applying K-means"
        @info "generating kluster"
        kmn = kmeans(x, clusters_fine; max_iters=10^6)
        X = kmn.assignments
        @info "writing data for $file"
        hfile = h5open("data/" * file * "_fine_cluster.hdf5", "w")
        hfile["X"] = X
        hfile["dt"] = dt
        close(hfile)
    end
end