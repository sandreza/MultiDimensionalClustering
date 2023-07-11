using MarkovChainHammer
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.Utils: autocovariance
using ParallelKMeans
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random, Plots
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer
using MultiDimensionalClustering.AlternativeGenerator
using Main.MarkovChainHammer.BayesianMatrix:BayesianGenerator
using Main.MarkovChainHammer.TransitionMatrix: steady_state
using Main.MarkovChainHammer.TransitionMatrix:perron_frobenius
using LsqFit
using SparseArrays
xₑₙ = [encoder(x̃[:,i])[1] for i in 1:1:564705]
xₑₙ = hcat(xₑₙ...)
##
X = copy(x̃) # copy(xₑₙ)
##
function split(X)
    numstates = 2
    r0 = kmeans(X, numstates; max_iters=10000)
    child_0 = (r0.assignments .== 1)
    child_1 = (!).(child_0)
    children = [view(X, :, child_0), view(X, :, child_1)]
    return r0.centers, children
end
level_global_indices(level) = 2^(level-1):2^level-1
##
levels = 11
parent_views = []
centers_list = Vector{Vector{Float64}}[]
push!(parent_views, X)
## Level 1
centers, children = split(X)
push!(centers_list, [centers[:, 1], centers[:, 2]])
push!(parent_views, children[1])
push!(parent_views, children[2])
## Levels 2 through levels
for level in ProgressBar(2:levels)
    for parent_global_index in level_global_indices(level)
        centers, children = split(parent_views[parent_global_index])
        push!(centers_list, [centers[:, 1], centers[:, 2]])
        push!(parent_views, children[1])
        push!(parent_views, children[2])
    end
end
@info "done with k-means"
##
struct StateTreeEmbedding{S, T}
    markov_states::S
    levels::T
end
function (embedding::StateTreeEmbedding)(current_state)
    global_index = 1 
    for level in 1:embedding.levels
        new_index = argmin([norm(current_state - markov_state) for markov_state in embedding.markov_states[global_index]])
        global_index = child_global_index(new_index, global_index)
    end
    return local_index(global_index, embedding.levels)
end

# assumes binary tree
local_index(global_index, levels) = global_index - 2^levels + 1 # markov index from [1, 2^levels]
# parent local index is markov_index(global_index, levels-1)
# child local index is 2*markov_index(global_index, levels-1) + new_index - 1
# global index is 2^levels + 1 + child local index
child_global_index(new_index, global_parent_index, level) = (2 * (local_index(global_parent_index, level - 1)-1) + new_index - 1) + 2^(level) 
# simplified:
child_global_index(new_index, global_parent_index) = 2 * global_parent_index + new_index - 1 
# global_indices per level
level_global_indices(level) = 2^(level-1):2^level-1
parent_global_index(child_index) = div(child_index, 2) # both global
centers_matrix = zeros(length(centers_list[1][1]), length(centers_list[1]), length(centers_list))
for i in eachindex(centers_list)
    centers_matrix[:, :, i] = hcat(centers_list[i]...)
end
centers_list = [[centers_matrix[:,1, i], centers_matrix[:,2, i]] for i in 1:size(centers_matrix)[3]]
# constructing embedding with 2^levels number of states
# note that we can also choose a number less than levels
levels = 11
embedding = StateTreeEmbedding(centers_list, levels)
##
@info "applying embedding"
markov_embedding = [embedding(X[:,i]) for i in 1:size(X)[2]]
@info "done with embedding"
##
Q = BayesianGenerator(markov_embedding)
##
Λ, V = eigen(Q)
W = inv(V)
##
P1 = exp(mean(Q))
q_min = 0.0
@info "leicht_newman_with_tree"
F, G, H, PI = leicht_newman_with_tree(P1, q_min)
@info "done with Leicht Newmann algorithm"
node_labels, adj, adj_mod, edge_numbers = graph_from_PI(PI)
X_LN = classes_timeseries(F, markov_embedding)
##
shift = 10000
skip = 5
inds = 1+shift:skip:12000+shift
colormap = :glasbey_hv_n256

L = 34 
dt =  0.017 * skip 
ts = collect(0:length(inds)-1) .* dt
xs = collect(0:63) / 64 * L 
Δx = xs[2] - xs[1]
energy = [sum(x[:, i] .^2 .* Δx) for i in 1:size(x)[2]] 
##

firstline = ts[230]
secondline = ts[700]
thirdline = ts[1825]
fourthline = ts[2100]
opacity = 0.5
linewidth = 10
linecolor = :yellow
linecolor2 = :black
labelsize = 40

axis_options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
axis2_options = (; titlesize = labelsize, xlabel = "space", ylabel = "time", ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution = (2120, 1432))
ax1 = Axis(fig[1,2]; title = "Coarse-Grained Cluster Dynamics", xlabel = "time", axis_options..., ylabel = "Cluster Label")
scatter!(ax1, ts, X_LN[inds]; color = X_LN[inds], colormap = colormap)
ax1.yticks = ([2, 4, 6, 8, 10, 12, 14])# ([-75, -50, -25, 0, 25, 50, 75], ["75S", "50S", "25S", "0", "25N", "50N", "75N"])
vlines!(ax1, firstline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, secondline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, thirdline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax1, fourthline, linewidth = 10, color = (linecolor2, opacity))
GLMakie.ylims!(0, 14)
ax2 = Axis(fig[2,2]; title = "Energy Dynamics", xlabel = "time", axis_options..., ylabel = "Energy")
lines!(ax2, ts, energy[inds], color = X_LN[inds], colormap = colormap)
vlines!(ax2, firstline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, secondline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, thirdline, linewidth = 10, color = (linecolor2, opacity))
vlines!(ax2, fourthline, linewidth = 10, color = (linecolor2, opacity))
#=
ax21 = Axis(fig[2,1])
scatter!(ax21, k2_t[inds])
ax22 = Axis(fig[2,2])
scatter!(ax22, k3_t[inds])
=#
ax3 = Axis(fig[1:2, 1]; title = "Space-Time Dynamics", axis2_options...)
GLMakie.heatmap!(ax3, xs, ts, x[:, inds], colormap = :balance, colorrange = (-3,3), interpolate = true)
hlines!(ax3, firstline, linewidth = linewidth, color = linecolor)
hlines!(ax3, secondline, linewidth = linewidth, color = linecolor)
hlines!(ax3, thirdline, linewidth = linewidth, color = linecolor)
hlines!(ax3, fourthline, linewidth = linewidth, color = linecolor)
display(fig)
##
Qred = mean(BayesianGenerator(X_LN))
##
Qpet = alternative_generator(Qred, X_LN, 1, 100) 
##
Qred 
eigen(Qred)
##
choice = 2
km = [real(W[end-choice, :])[me] for me in markov_embedding]
##
km1decay = autocovariance(km; timesteps= 10000, progress = true)
scatter(km1decay)
##

if abs(imag(Λ[end-choice])) > eps(1e6)
    pguess = [real(Λ[end-choice]), imag(Λ[end-choice])]
    modeli = model2(x, p) = @. exp(x * p[1]) * cos(x * p[2])
else
    pguess = [real(Λ[end-choice])]
    modeli = model1(x, p) = @. exp(x * p[1])
end
ydata = copy(km1decay) / km1decay[1]
tmp = curve_fit(modeli, 1:length(ydata), ydata, pguess)
tmp.param
pguess
##
fig = Figure() 
ax = Axis(fig[1,1])
scatter!(ax, 1:length(ydata), ydata, color = :black)
lines!(ax, 1:length(ydata), model1(1:length(ydata), tmp.param), color = :red)
display(fig)
##
kms = [[real(W[end-choice, :])[me] for me in markov_embedding] for choice in 1:4]
##
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, x̃[1, inds], color = :black)
ax2 = Axis(fig[1,2])
lines!(ax2, kms[1][inds], color = :red)
ax21 = Axis(fig[2,1])
lines!(ax21, mean(x̃[:, inds] .^2, dims = 1)[:], color = :black)
ax22 = Axis(fig[2,2])
lines!(ax22, kms[2][inds], color = :red)
display(fig)
##
function grab_states(centers_list)
    nstates = length(centers_list)+1
    m = div(nstates, 2)
    states = typeof(centers_list[1][1])[]
    for i in 1:m
        push!(states, centers_list[i+m-1][1])
        push!(states, centers_list[i+m-1][1])
    end
    return states
end
##
EQ = eigen(Q)
##
mstates = grab_states(centers_list)
##
g⃗ = norm.(decoder.(mstates))
##
gt = sqrt.(sum(x̃ .^2, dims = 1)[:]) 
#
ct = autocovariance(gt; timesteps= 10000, progress = true)
##
ce = autocovariance(g⃗, EQ, 1:10000; progress=true) 
##
field = real.(sum(V[:, end-1]  .* decoder.(mstates))) 
field2 = real.(sum(V[:, end-2]  .* decoder.(mstates))) 
scatter(field)
scatter!(field2)
##
markov_embedding
p = real.(V[:, end] / sum(V[:,end]))
statemax = argmax(p)
##
indchoice = 512
conditional = Vector{Float64}[]
for i in ProgressBar(1:size(x̃)[2])
    if markov_embedding[i] == indchoice
        push!(conditional, x̃[:, i])
    end
end

# obs(x) = x[1]
# hist(obs.(conditional))
fig = Figure() 
ax = Axis(fig[1,1])
for i in 1:10:length(conditional)
    scatter!(ax, conditional[i])
end
lines!(ax, mstates[indchoice], color = :black, linewidth = 4)
display(fig)