using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering, NetworkLayout, SparseArrays, Graphs, GraphMakie, Printf
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/potential_well.hdf5")
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
τ = dt
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

transparancy = 0.4 * tanh.(adj_mod.nzval[:] / adj_mod.nzval[3]) .+ 0.1
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
           nlabels_fontsize=nlabels_fontsize)
hidedecorations!(ax);
hidespines!(ax);
display(fig)