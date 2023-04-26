using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering, NetworkLayout, SparseArrays, Graphs, GraphMakie, Printf

fig = Figure(resolution=(1707, 885))
Random.seed!(12345)
set_theme!(backgroundcolor=:white)
fig = Figure(resolution=(2561, 1027))
files = ["potential_well", "lorenz", "newton"]
for (i, file) in enumerate(files)
    hfile = h5open("data/" * file * ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)

    @info "opening fine partition"
    hfile = h5open("data/" * file * "_fine_cluster.hdf5")
    X = read(hfile["X"])
    close(hfile)

    @info "opening coarse partition"
    hfile = h5open("data/"* file *"_coarse_cluster.hdf5")
    edge_numbers = read(hfile["tree_edge_numbers"])
    N = read(hfile["tree_matrix_size"])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in 1:edge_numbers
        ii = read(hfile["tree_i"*string(i)])
        jj = read(hfile["tree_j"*string(i)])
        modularity_value = read(hfile["tree_modularity"*string(i)])
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end
    layout = Buchheim()
    node_labels = zeros(Float64, N)
    for i in 1:edge_numbers
        ii = read(hfile["tree_i"*string(i)])
        modularity_value = read(hfile["tree_modularity"*string(i)])
        node_labels[ii] = modularity_value
    end
    F = Vector{Int64}[]
    Flength = read(hfile["Flength"])
    for i in 1:Flength
        push!(F, read(hfile["F"*string(i)]))
    end
    X_LN = classes_timeseries(F, X)
    close(hfile)
    ##
    @info "plotting"
    # https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
    # for colorschemes
    ax_old = LScene(fig[i, 1]; show_axis=false) # Axis3(fig[1, 1]; perspectiveness = 1)# 
    ax_middle = Axis(fig[i, 2])  # Axis(fig[1,2]) # 
    ax_new = LScene(fig[i, 3]; show_axis=false) # Axis3(fig[1,3])# 
    sc = lines!(ax_old, x[1, :], x[2, :], x[3, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
    lines!(ax_new, x[1, :], x[2, :], x[3, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)

    rotate_cam!(ax_old.scene, (0.0, -10.5, 0.0))
    rotate_cam!(ax_new.scene, (0.0, -10.5, 0.0))

    G = SimpleDiGraph(adj)
    transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
    nlabels_fontsize = 25
    edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
    nlabels = [@sprintf("%.0e", node_labels[i]) for i in 1:nv(G)]
    graphplot!(ax_middle, G, layout=layout, nlabels=nlabels, node_size=100,
        node_color=(:orange, 0.9), edge_color=edge_color, edge_width=3,
        arrow_size=25, nlabels_align=(:center, :center),
        nlabels_fontsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
    hidedecorations!(ax_middle)
    hidespines!(ax_middle)

    display(fig)

end
display(fig)