using HDF5, GLMakie, GraphMakie, Statistics
using MarkovChainHammer.BayesianMatrix

include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, LaTeXStrings, GLMakie
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs
using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, ProgressBars, InvertedIndices
using MultiDimensionalClustering.CommunityDetection, ProgressBars
##
function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end

##
# seems incorrect. Not identifying the correct coarse cluster

file = "potential_well"
temp1, temp2, temp3, temp4, temp5, temp6 = read_coarse_cluster_tree(file)
X_LN_array1 = temp1
adj_array = temp2
adj_mod_array = temp3
node_labels_array = temp4
edge_numbers_array = temp5
τ = temp6

X, Xc = read_fine_cluster(file)
x, dt = read_data(file)

# Somehow incorrect
temp1, temp2, temp3, temp4, temp5, temp6, temp7 = read_coarse_cluster(file)
X_LN_array2 = temp1
Q_array = temp2
Q_pert_array = temp3
Pt_array = temp4
Qt_array = temp5
Qt_pert_array = temp6
##
Q = mean(BayesianGenerator(X; dt=dt))
Λ = eigvals(Q)
X_LN_array1 = []
Q_array = []
PIs = []
for i in ProgressBar(1:8)
    q_min = 1e-16
    τ = 1.0 / -real(Λ[end-i])
    P = exp(Q*τ)
    F, G, H, PI = leicht_newman_with_tree(P, q_min)
    X_LN = classes_timeseries(F, X)
    Qtmp = mean(BayesianGenerator(X_LN; dt=dt))
    push!(X_LN_array1, X_LN)
    push!(Q_array, Qtmp)
    push!(PIs, PI)
end

##
# resolution=(5000, 4000)
n_timescales = length(τ)
t = 1

## 
# Create Adjacency matrix 
PI = PIs[t]
N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
adj = spzeros(Int64, N, N)
adj_mod = spzeros(Float64, N, N)
for i in ProgressBar(eachindex(PI))
    ii = PI[i][1]
    jj = PI[i][2]
    modularity_value = PI[i][3]
    adj[ii, jj] += 1
    adj_mod[ii, jj] = modularity_value
end
adj_array = [adj]
adj_mod_array = [adj_mod]
edge_numbers_array = [length(PI)]

##
fig = Figure(resolution = (3200, 800))
layout = Buchheim()
colormap = :glasbey_hv_n256
set_theme!(backgroundcolor=:white)

ax11 = Axis(fig[1,2]; title =  @sprintf("τ = %5.2f", 1.0 / -real(Λ[end-t])), titlesize = 40)
G = SimpleDiGraph(adj_array[t])
transparancy = 0.4 * adj_mod_array[t].nzval[:] / adj_mod_array[t].nzval[1] .+ 0.1
nlabels_fontsize = 35
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers_array[t]]
nlabels = [@sprintf("%2.2f", node_labels_array[t][i]) for i in 1:nv(G)]
graphplot!(ax11, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=5,
    arrow_size=45, nlabels_align=(:center, :center),
    nlabels_textsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
# cc = cameracontrols(ax11.scene)
hidedecorations!(ax11)
hidespines!(ax11);
ax12 = LScene(fig[1,3]; show_axis = false)
res = 10
markersize = 10.0
scatter!(ax12, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=cgrad(colormap)[X_LN_array1[t][1:res:end]])
# rotate_cam!(ax12.scene, (0.0, -10.5, 0.0))

ax13 = LScene(fig[1,1]; show_axis = false)
res = 10
markersize = 10.0
scatter!(ax13, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], markersize=markersize, color=X[1:res:end], colormap = colormap)
# rotate_cam!(ax13.scene, (0.0, -10.5, 0.0))

display(fig)

ax = Axis(fig[1,4])
Q_graph = mean(BayesianGenerator(X_LN_array1[t], dt=dt))# Q_array[ii]
g_Q = DiGraph(Q_graph')
Q_prim = zeros(size(Q_graph))
for i in 1:size(Q_graph)[1]
    Q_prim[:, i] .= -Q_graph[:, i] / Q_graph[i, i]
    Q_prim[i, i] = -1 / Q_graph[i, i]
end
elabels = string.([round(Q_prim[i]; digits=2) for i in 1:ne(g_Q)])
# [@sprintf("%.0e", node_labels[i]) for i in 1:nv(G)]
transparancy = [Q_prim[i] for i in 1:ne(g_Q)]
elabels_color = [(:black, transparancy[i] > eps(100.0)) for i in 1:ne(g_Q)]
#edge_color_Q = [(:black, transparancy[i]) for i in 1:ne(g_Q)]
edge_color_Q = [(cgrad(colormap)[(i-1)÷(size(Q_graph)[1])+1], transparancy[i]) for i in 1:ne(g_Q)]
node_color = [(cgrad(colormap)[i]) for i in 1:nv(g_Q)]
edge_attr = (; linestyle=[:dot, :dash, :dash, :dash, :dot, :dash, :dash, :dash, :dot])
elabels_fontsize = 40
nlabels_fontsize = 40
node_size = 100.0
edge_width_Q = [10.0 for i in 1:ne(g_Q)]
arrow_size_Q = [40.0 for i in 1:ne(g_Q)]
node_labels_Q = repr.(1:nv(g_Q))
kwargs_edges  = (;  elabels_color=elabels_color, elabels_textsize=elabels_fontsize, edge_color=edge_color_Q, edge_width=edge_width_Q, elabels=elabels)
kwargs_nodes  = (; node_color=node_color, node_size=node_size, nlabels=node_labels_Q, nlabels_textsize=nlabels_fontsize)
kwargs_arrows = (; arrow_size=arrow_size_Q)
graphplot!(ax, g_Q; kwargs_edges..., kwargs_nodes..., kwargs_arrows..., layout = Stress())
hidedecorations!(ax)
hidespines!(ax)
display(fig)

