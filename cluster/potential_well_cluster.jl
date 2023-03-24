using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/potential_well.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
##
@info "applying K-means"
kmn = kmeans(x, 200; max_iters=10^6);
X = kmn.assignments
##
@info "Computing Generator and Eigenvalue Decomposition"
Q = mean(BayesianGenerator(X; dt = dt))
p =  steady_state(Q)
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
Qnew = (Q + Diagonal(p) * Q' * Diagonal(1 ./ p)) * 0.5 # remove imaginary eigenvalues
Λ, V = eigen(Qnew)
##
@info "applying Modified Leicht-Newman clustering"
cluster_heuristic = 10
τ = dt # 1.0  / -real(Λ[end + 1 - cluster_heuristic]) # the relevant timescales are proportional to the inverse eigenvalues
q_min = sqrt(eps(1.0)) # 1e-16 
P = exp(Qnew * τ)
F, G, H = leicht_newman(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN = classes_timeseries(F, X)
##
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
set_theme!(backgroundcolor=:black)
fig = Figure(resolution=(1707, 885))
ax_old = LScene(fig[1,1]; show_axis = false)
ax_new = LScene(fig[1, 2]; show_axis=false)
scatter!(ax_old, x[1, :], x[2, :], x[3, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
scatter!(ax_new, x[1, :], x[2, :], x[3, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
display(fig)
##
# Koopman Modes
timesteps = 4000
W = inv(V)
gs = Vector{Float64}[]
for index in [2, 3, 4]
    g = W[end-index+1, :]
    g = [real(g[x]) for x in X[1:timesteps]]
    push!(gs, g)
end
##
set_theme!(backgroundcolor=:white)
fig_trajectory = Figure()
ax1 = Axis(fig_trajectory[1, 1]; title = "fastest")
ax2 = Axis(fig_trajectory[2, 1]; title = "medium")
ax3 = Axis(fig_trajectory[3, 1]; title = "slowest")
scatter!(ax1, x[1,1:timesteps], color = X_LN[1:timesteps], colormap = :glasbey_hv_n256, markersize = 20.0, markerspacing = 0.1, markerstrokewidth = 0.0)
scatter!(ax2, x[2,1:timesteps], color = X_LN[1:timesteps], colormap = :glasbey_hv_n256, markersize = 20.0, markerspacing = 0.1, markerstrokewidth = 0.0)
scatter!(ax3, x[3,1:timesteps], color = X_LN[1:timesteps], colormap = :glasbey_hv_n256, markersize = 20.0, markerspacing = 0.1, markerstrokewidth = 0.0)
for i in 1:3
    ax = Axis(fig_trajectory[i, 2]; title = "Koopman Mode $(3-i + 1)")
    scatter!(ax, gs[3-i + 1], color=X_LN[1:timesteps], colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
end
display(fig_trajectory)
##
# number of clusters a function of qmin and τ
τs = [dt, 1.0 / -real(Λ[end-3]), 1.0 / -real(Λ[end-2]), 1.0 / -real(Λ[end-1])]
noc = Vector{Int64}[]
qmins = [10.0^i for i in -6:0.05:3]

for τ in ProgressBar(τs)
    P = exp(Qnew * τ)
    number_of_clusters = Int64[]
    for qmin in ProgressBar(qmins)
        F, G, H = leicht_newman(P, qmin)
        push!(number_of_clusters, length(F))
    end
    push!(noc, number_of_clusters)
end
##
τ_labels = ["τ = dt", "τ = -1 / λ₄", "τ = -1 / λ₃", "τ = -1 / λ₂"]
labelsize = 40
axis_options = (; xlabel="log10(qmin)", ylabel="Number of Clusters", xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)

fig = Figure(resolution=(2040, 1206))
for i in 1:4
    jj = (i-1) % 2 + 1
    ii = (i-1) ÷ 2 + 1
    ax = Axis(fig[ii, jj]; title = τ_labels[i], axis_options...)
    scatter!(ax, log10.(qmins),  noc[i], markersize = 20.0, markerspacing = 0.1, markerstrokewidth = 0.0)
    xlims!(ax, -7, 4)
    ax.xticks = (collect(-7:4), string.(collect(-7:4)))
    ax.yticks = (collect(1:8), string.(collect(1:8))) 
    ylims!(ax, 0, 9)
end
display(fig)