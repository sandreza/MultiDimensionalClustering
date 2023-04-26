using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
Random.seed!(12345)
check_file = false

files = ["potential_well", "lorenz", "newton"]
cluster_heuristics = [8, 2, 8]
for (i, file) in ProgressBar(enumerate(files))
    @info "opening data for $file"
    hfile = h5open("data/" * file * "_fine_cluster.hdf5")
    X = read(hfile["X"])
    dt = read(hfile["dt"])
    close(hfile)
    clustername = "data/" * file * "_coarse_cluster.hdf5"
    if isfile(pwd() * clustername) && check_file
        @info "data already exists. skipping data generation"
    else
        @info "constructing generator"
        Q = mean(BayesianGenerator(X; dt=dt))
        @info "computing eigenvalue decomposition"
        Λ, V = eigen(Q)
        @info "applying leicht_newman"
        cluster_heuristic = cluster_heuristics[i]
        τ = 1.0 / -real(Λ[end+1-cluster_heuristic]) # the relevant timescales are proportional to the inverse eigenvalues, note, can't be less than Δt by construction
        q_min = 1e-16
        P = exp(Q * τ)
        F, G, H, PI = leicht_newman_with_tree(P, q_min)
        @info "writing data for $file"
        hfile = h5open(clustername, "w")
        for i in eachindex(F)
            hfile["F"*string(i)] = F[i]
            hfile["G"*string(i)] = G[i]
        end
        hfile["eigenvalues"] = Λ
        hfile["eigenvectors"] = V
        hfile["generator"] = Q
        hfile["tree_edge_numbers"] = length(PI)
        hfile["Flength"] = length(F)
        N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
        hfile["tree_matrix_size"] = N
        for i in eachindex(PI)
            hfile["tree_i"*string(i)] = PI[i][1]
            hfile["tree_j"*string(i)] = PI[i][2]
            hfile["tree_modularity"*string(i)] = PI[i][3]
        end
        close(hfile)
    end
end