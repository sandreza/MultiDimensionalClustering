using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random, Plots
include("/Users/ludovico/Downloads/MarkovChainHammer.jl-main/src/MarkovChainHammer.jl")
using MultiDimensionalClustering.CommunityDetection
using Main.MarkovChainHammer.BayesianMatrix:BayesianGenerator
using Main.MarkovChainHammer.TransitionMatrix: steady_state
using Main.MarkovChainHammer.TransitionMatrix:perron_frobenius
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/PIV.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
x = x[:,1:10:end]
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

x[3,:]
##
@info "applying Modified Leicht-Newman clustering"
cluster_heuristic = 2
τ = 1.0 / -real(Λ[end+1-cluster_heuristic]) # the relevant timescales are proportional to the inverse eigenvalues, note, can't be less than Δt by construction
q_min = 0.
P = exp(Q * 10*dt)
F, G, H = leicht_newman(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN = classes_timeseries(F, X)
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
# fig = Figure(resolution=(1707, 885))
# ax_old = LScene(fig[1, 1]; show_axis=false)
# ax_new = LScene(fig[1, 2]; show_axis=false)
# set_theme!(backgroundcolor=:black)
# lines!(ax_old, x[2, :], x[3, :], x[4, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
# lines!(ax_new, x[2, :], x[3, :], x[4, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
# display(fig)
##
P = perron_frobenius(X,1)
F, G, H = leicht_newman(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN1 = classes_timeseries(F, X)
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
# fig = Figure(resolution=(1707, 885))
# ax_old = LScene(fig[1, 1]; show_axis=false)
# ax_new = LScene(fig[1, 2]; show_axis=false)
# set_theme!(backgroundcolor=:black)
# lines!(ax_old, x[2, :], x[3, :], x[4, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
# lines!(ax_new, x[2, :], x[3, :], x[4, :], color=X_LN1, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
# display(fig)
##
fig = Figure(resolution=(1707, 885))
ax_old = LScene(fig[1, 1]; show_axis=false)
ax_new = LScene(fig[1, 2]; show_axis=false)
set_theme!(backgroundcolor=:black)
lines!(ax_old, x[2, :], x[3, :], x[4, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax_new, x[2, :], x[3, :], x[4, :], color=X_LN1, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
display(fig)

##
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random, Plots
include("/Users/ludovico/Downloads/MarkovChainHammer.jl-main/src/MarkovChainHammer.jl")
using MultiDimensionalClustering.CommunityDetection
using Main.MarkovChainHammer.BayesianMatrix:BayesianGenerator
using Main.MarkovChainHammer.TransitionMatrix: steady_state
using Main.MarkovChainHammer.TransitionMatrix:perron_frobenius
Random.seed!(12345)
##
@info "opening data"
hfile = h5open("data/PIV.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
close(hfile)
x = x[1:32,1:1:end]
##
@info "applying K-means"
kmn = kmeans(x, 1000; n_threads=12, max_iters=10^6);
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
t_steps = 10
q_min = 0.
PQ = exp(Q*t_steps*dt)
F, G, H = leicht_newman(PQ, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN_Q = classes_timeseries(F, X)

P = perron_frobenius(X,t_steps)
F, G, H = leicht_newman(P, q_min)
@info "The algorithm identified $(length(F)) clusters"
X_LN = classes_timeseries(F, X)
##
fig = Figure(resolution=(1707, 885))
ax1 = LScene(fig[1, 1]; show_axis=false)
ax2 = LScene(fig[1, 2]; show_axis=false)
ax3 = LScene(fig[1, 3]; show_axis=false)
set_theme!(backgroundcolor=:white)
lines!(ax1, x[2, :], x[3, :], x[4, :], color=X, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax2, x[2, :], x[3, :], x[4, :], color=X_LN_Q, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
lines!(ax3, x[2, :], x[3, :], x[4, :], color=X_LN, colormap=:glasbey_hv_n256, markersize=20.0, markerspacing=0.1, markerstrokewidth=0.0)
display(fig)