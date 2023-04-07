using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering, NetworkLayout, SparseArrays, Graphs, GraphMakie, Printf
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/lorenz.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
##
@info "applying K-means"
kmn = kmeans(x, 1200; max_iters=10^6);
X = kmn.assignments
##
@info "Computing Generator and Eigenvalue Decomposition"
Q = mean(BayesianGenerator(X; dt=dt))
p = steady_state(Q)
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
Λ, V = eigen(Q)
##
@info "applying Modified Leicht-Newman clustering"
τ = -1 ./ real(Λ[end-50])
q_min = sqrt(eps(1.0)) # 1e-16 
P = exp(Q * τ)
F, G, H, PI = leicht_newman_with_tree(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN = classes_timeseries(F, X)
##
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
##
G = SimpleDiGraph(adj)
fig = Figure(resolution=(1200, 1000))
ax = Axis(fig[1, 1])
nlabels_fontsize = 25
nlabels = [@sprintf("%.0e",node_labels[i]) for i in 1:nv(G)]
graphplot!(ax, G, layout=layout, nlabels = nlabels, node_size = 100, 
           node_color = (:orange, 0.9), edge_color = edge_color, edge_width = 3,
            arrow_size=25, nlabels_align=(:center, :center),
           nlabels_fontsize=nlabels_fontsize, tangents=((0,-1),(0,-1)))
hidedecorations!(ax);
hidespines!(ax);
display(fig)

##
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
set_theme!(backgroundcolor=:white)
fig = Figure(resolution=(2561, 1027))
ax_old = LScene(fig[1, 1]; show_axis=false) # Axis3(fig[1, 1]; perspectiveness = 1)# 
ax_middle = Axis(fig[1, 2])  # Axis(fig[1,2]) # 
ax_new = LScene(fig[1, 3]; show_axis=false) # Axis3(fig[1,3])# 
sc = lines!(ax_old, x[1, :], x[2, :], x[3, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax_new, x[1, :], x[2, :], x[3, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)

rotate_cam!(ax_old.scene, (0.0, -10.5, 0.0))
rotate_cam!(ax_new.scene, (0., -10.5, 0.))
edge_color = [(:red, transparancy[i]) for i in eachindex(PI)]
graphplot!(ax_middle, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=3,
    arrow_size=25, nlabels_align=(:center, :center),
    nlabels_fontsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
hidedecorations!(ax_middle);
hidespines!(ax_middle);

display(fig)