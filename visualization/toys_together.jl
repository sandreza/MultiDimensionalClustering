using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering, NetworkLayout, SparseArrays, Graphs, GraphMakie, Printf

##
fig = Figure(resolution=(1707, 885))
Random.seed!(12345)
files = ["potential_well", "lorenz", "newton"]
for (i, file) in enumerate(files)
    @info "opening data for $file"
    hfile = h5open("data/" * file * ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    @info "applying K-means"
    kmn = kmeans(x, 1000; max_iters=10^6)
    X = kmn.assignments
    @info "Computing Generator and Eigenvalue Decomposition"
    Q = mean(BayesianGenerator(X; dt=dt))
    p = steady_state(Q)
    Λ, V = eigen(Q)
    @info "applying Modified Leicht-Newman clustering"
    τ = real(-1 / Λ[end-4])# dt
    q_min = sqrt(eps(1.0)) # 1e-16 
    P = exp(Q * τ)
    F, G, H, PI = leicht_newman_with_tree(P, q_min)
    @info "The algorithm identified $(length(F)) clusters"
    X_LN = classes_timeseries(F, X)
    kmn_reduced = kmeans(x, maximum(X_LN); max_iters=10^6)
    X_KM = kmn_reduced.assignments
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in eachindex(PI)
        adj[PI[i][1], PI[i][2]] += 1
        adj_mod[PI[i][2], PI[i][1]] = PI[i][3]
    end
    layout = Buchheim()
    node_labels = zeros(Float64, N)
    for i in eachindex(PI)
        node_labels[PI[i][1]] = PI[i][3]
    end
    transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
    edge_color = [(:red, transparancy[i]) for i in eachindex(PI)]
    G = SimpleDiGraph(adj)
    nlabels_fontsize = 25
    nlabels = [@sprintf("%.0e", node_labels[i]) for i in 1:nv(G)]

    ax_old = LScene(fig[i, 2]; show_axis=false)
    ax_old2 = LScene(fig[i, 1]; show_axis=false)
    ax_new = LScene(fig[i, 4]; show_axis=false)
    ax_LN = Axis(fig[i, 3])
    scatter!(ax_old, x[1, :], x[2, :], x[3, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
    scatter!(ax_old2, x[1, :], x[2, :], x[3, :], color=X_KM, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
    graphplot!(ax_LN, G, layout=layout, nlabels=nlabels, node_size=100,
        node_color=(:orange, 0.9), edge_color=edge_color, edge_width=3,
        arrow_size=25, nlabels_align=(:center, :center),
        nlabels_fontsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
    hidedecorations!(ax_LN)
    hidespines!(ax_LN)
    scatter!(ax_new, x[1, :], x[2, :], x[3, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
end

display(fig)