include("GenerateFineCluster.jl")
include("GenerateCoarseClusterTree.jl")
include("GenerateCoarseCluster.jl")

using Main.GenerateFineCluster:read_fine_cluster, plot_fine_cluster, plot_fine_cluster_array, fine_cluster
using Main.GenerateCoarseClusterTree:read_coarse_cluster_tree, plot_coarse_cluster_tree, coarse_cluster_tree
using Main.GenerateCoarseCluster:coarse_cluster, plot_coarse_cluster1, plot_coarse_cluster2, read_coarse_cluster
using HDF5, LaTeXStrings, GLMakie
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs

##
function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end

function plot_fine_clusterP(x,X,title,figure_number,mks,res,azimuth,elevation)
    fig = Figure(resolution=(5000, 3000))
    colormap = :glasbey_hv_n256
    for i in 1:3
        ax = Axis3(fig[1, 2*(i-1)+1:2*i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth[i], elevation=elevation[i])
        scatter!(ax, x[i][1, 1:res[i]:end], x[i][2, 1:res[i]:end], x[i][3, 1:res[i]:end], color=cgrad(colormap)[(X[i][1:res[i]:end] .% 256 .+1)]; markersize=mks[i])
        Label(fig[1, 2*(i-1)+1:2*i, Top()], title[i]; textsize=100)
    end
    for i in 1:2
        ax = Axis3(fig[2, 3*(i-1)+1:3*i], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth[i+3], elevation=elevation[i+3])
        scatter!(ax, x[i+3][1, 1:res[i+3]:end], x[i+3][2, 1:res[i+3]:end], x[i+3][3, 1:res[i+3]:end], color=cgrad(colormap)[(X[i+3][1:res[i+3]:end] .% 256 .+1)]; markersize=mks[i+3])
        Label(fig[2, 3*(i-1)+1:3*i, Top()], title[i+3]; textsize=100)
    end
    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

x = []
dt = []
files = ["potential_well", "lorenz", "newton", "kuramoto", "PIV"]
for i in 1:5
    temp1, temp2 = read_data(files[i])
    push!(x, temp1)
    push!(dt, temp2)
end

X = []
Xc = []
for i in (1:5)
    temp1, temp2 = read_fine_cluster(files[i])
    push!(X, temp1)
    push!(Xc, temp2)
end

X_LN_array1 = []
adj_array = []
adj_mod_array = []
node_labels_array = []
edge_numbers_array = []
τ = []
for i in 1:5
    temp1, temp2, temp3, temp4, temp5, temp6 = read_coarse_cluster_tree(files[i])
    push!(X_LN_array1, temp1)
    push!(adj_array, temp2)
    push!(adj_mod_array, temp3)
    push!(node_labels_array, temp4)
    push!(edge_numbers_array, temp5)
    push!(τ, temp6)
end

X_LN_array2 = []
Q_array = []
Q_pert_array = []
Pt_array = []
Qt_array = []
Qt_pert_array = []
score = []
for i in 1:5
    temp1, temp2, temp3, temp4, temp5, temp6, temp7 = read_coarse_cluster(files[i])
    push!(X_LN_array2, temp1)
    push!(Q_array, temp2)
    push!(Q_pert_array, temp3)
    push!(Pt_array, temp4)
    push!(Qt_array, temp5)
    push!(Qt_pert_array, temp6)
    push!(score, temp7)
end
##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]
mks = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.1pi]
elevation = [0.2pi, 0.2pi, 0.2pi, 0.1pi, 0.1pi]
res = [1 ,1 ,1 ,1 ,1]

# arr_fig = [1,2,3]
# figure_number = [1 1]
# plot_fine_cluster_array(x[arr_fig],X[arr_fig], titles[arr_fig], figure_number, mks[arr_fig], res[arr_fig], azimuth[arr_fig], elevation[arr_fig])

# arr_fig = [4,5]
# figure_number = [1 2]
# plot_fine_cluster_array(x[arr_fig],X[arr_fig], titles[arr_fig], figure_number, mks[arr_fig], res[arr_fig], azimuth[arr_fig], elevation[arr_fig])

figure_number = [1 1]
plot_fine_clusterP(x,X, titles, figure_number, mks, res, azimuth, elevation)

##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]

mks_vals = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.05pi]
elevation = [0.2pi, 0.2pi, 0.2pi, 0.1pi, 0.2pi]
for f_index in 1:5
    figure_number = [2 f_index]
    plot_coarse_cluster_tree(x[f_index],X_LN_array1[f_index], adj_array[f_index], adj_mod_array[f_index], 
                        node_labels_array[f_index], edge_numbers_array[f_index], τ[f_index], titles[f_index], figure_number; res = 1, mks = mks_vals[f_index], azimuth=azimuth[f_index], elevation=elevation[f_index])
end
##
titles = [latexstring("\\textbf{\\textrm{Multi-well potential}}") 
        latexstring("\\textbf{\\textrm{Lorenz 63}}") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}}") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}}") 
        latexstring("\\textbf{\\textrm{PIV}}")]

mks_vals = [5,5,3,3,20]
azimuth = [0.2pi, 0.05pi, 0.4pi, 1.2pi, 0.05pi]
elevation = [0.2pi, 0.2pi, 0.2pi, 0.1pi, 0.2pi]
for f_index in 1:5
    figure_number = [3 f_index]
    plot_coarse_cluster1(x[f_index], τ[f_index], X_LN_array2[f_index], Q_array[f_index], score[f_index], titles[f_index], figure_number; res = 1, mks = mks_vals[f_index], azimuth=azimuth[f_index], elevation=elevation[f_index])
end
##
indices = [2,3,3,7,3]
for f_index in 1:5
    titles = [latexstring("\\textbf{\\textrm{Multi-well potential}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Lorenz 63}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Newton-Leipnik}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{Kuramoto-Sivashinsky}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})") 
        latexstring("\\textbf{\\textrm{PIV}} \\;(\\textbf{t} = \\textbf{$(round(τ[f_index][indices[f_index]],sigdigits=3))})")]
    figure_number = [4 f_index]
    plot_coarse_cluster2(dt[f_index], Pt_array[f_index], Qt_array[f_index], Qt_pert_array[f_index], indices[f_index], titles[f_index], figure_number)
end
