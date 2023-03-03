using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/lorenz.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
##
@info "applying K-means"
kmn = kmeans(x, 1000; max_iters=10^6);
X = kmn.assignments
##
@info "Computing Generator and Eigenvalue Decomposition"
Q = mean(BayesianGenerator(X; dt=dt))
p = steady_state(Q)
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
Qnew = (Q + Diagonal(p) * Q' * Diagonal(1 ./ p)) * 0.5 # remove imaginary eigenvalues
Λ, V = eigen(Q)
##
@info "applying Modified Leicht-Newman clustering"
cluster_heuristic = 2
τ = 1.0 / -real(Λ[end+1-cluster_heuristic]) # the relevant timescales are proportional to the inverse eigenvalues, note, can't be less than Δt by construction
q_min = 1e-16
P = exp(Q * τ)
F, G, H = leicht_newman(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN = classes_timeseries(F, X)
##
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
fig = Figure(resolution=(1707, 885))
ax_old = LScene(fig[1, 1]; show_axis=false)
ax_new = LScene(fig[1, 2]; show_axis=false)
set_theme!(backgroundcolor=:black)
lines!(ax_old, x[1, :], x[2, :], x[3, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax_new, x[1, :], x[2, :], x[3, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
display(fig)
##
fig_trajectory = Figure()
ax1 = Axis(fig_trajectory[1, 1]; title="x")
ax2 = Axis(fig_trajectory[2, 1]; title="y")
ax3 = Axis(fig_trajectory[3, 1]; title="z")
timesteps = 4000
lines!(ax1, x[1, 1:timesteps], color=X_LN[1:timesteps], colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax2, x[2, 1:timesteps], color=X_LN[1:timesteps], colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax3, x[3, 1:timesteps], color=X_LN[1:timesteps], colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
display(fig_trajectory)