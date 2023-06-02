######## Just some experiments #######


using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using Combinatorics
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering.AlternativeGenerator
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs, InvertedIndices, LaTeXStrings

using LinearAlgebra, ProgressBars
using MarkovChainHammer.TransitionMatrix: perron_frobenius
export leicht_newman, classes_timeseries, greatest_common_cluster, leicht_newman_with_tree

function modularity_matrix(A)
    N = size(A)[1]
    b = zeros(N, N)
    for i = 1:N, j = 1:N
        b[i, j] = (A[i, j] - (1 - A[i, i]) * (sum(A[j, :])) / N) / N
    end
    B = Symmetric(b + b')
    return B
end

function principal_vector(B::Symmetric)
    s = ones(Int, size(B)[1])
    Λ, V = eigen(B)
    v₁ = V[:, sortperm(real.(Λ))[end]]
    s[v₁.<=0] .= -1
    return s
end

function modularity(B, s)
    return s' * (B * s)
end

function modularity_eig(B)
    Λ, V = eigen(B)
    return maximum(Λ)
end

modularity(B::Symmetric) = modularity(B, principal_vector(B))

function split_community(B, indices, q_min)
    Bg = B[indices, :][:, indices]
    Bg = Bg - Diagonal(sum(Bg, dims=1)[:])
    Bg = Symmetric(Bg + Bg')
    s = principal_vector(Bg)
    q = modularity(Bg)
    qq = q_min

    if (q > q_min)
        ind1 = [i for (j, i) in enumerate(indices) if s[j] == 1]
        ind2 = [i for (j, i) in enumerate(indices) if s[j] == -1]
        qq = q
        return ind1, ind2, qq
    end
    return [], [], qq
end

function leicht_newman_with_treeP(A, q_min::Float64)
    B = modularity_matrix(A)
    n = size(A)[1]
    W, F, G, P1, P2 = [collect(1:n)], [], [], [1], []
    qOld = 0.0
    H = []
    global_index = 1
    while (length(W) > 0)
        w = popfirst!(W)
        p1 = popfirst!(P1)
        ind1, ind2, q = split_community(B, w, q_min)
        if (length(ind1) > 0) & (length(ind2) > 0)
            W = [ind1, ind2, W...]
            P1 = [global_index + 1, global_index + 2, P1...]
            P2 = push!(P2, (p1, global_index + 1, q))
            P2 = push!(P2, (p1, global_index + 2, q))
            global_index += 2
            push!(H, [ind1, ind2, q])
            if q > 0
                qOld = q
            end
        else
            push!(F, w)
            push!(G, qOld)
        end
    end
    return F, G, H, P2
end

function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
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

##
f_index = 2
Q = mean(BayesianGenerator(X[f_index]; dt=dt[f_index]))
Λ, V = eigen(Q)
q_min = 1e-16
P = perron_frobenius(X[f_index],step=100)
F, G, H, PI = leicht_newman_with_tree(P, q_min)
N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])

adj = spzeros(Int64, N, N)
adj_mod = spzeros(Float64, N, N)
edge_numbers = length(PI)
for i in 1:edge_numbers
    ii = PI[i][1]
    jj = PI[i][2]
    modularity_value = PI[i][3]
    adj[ii, jj] += 1
    adj_mod[ii, jj] = modularity_value
end
layout = Buchheim()
node_labels = zeros(Float64, N)
for i in 1:edge_numbers
    ii = PI[i][1]
    modularity_value = PI[i][3]
    node_labels[ii] = modularity_value
end
##
@info "plotting"
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
# for colorschemes
set_theme!(backgroundcolor=:white)
fig = Figure(resolution=(2561, 1027))
ax_middle = Axis(fig[1, 1])  # Axis(fig[1,2]) # 

G = SimpleDiGraph(adj)
transparancy = 0.4 * adj_mod.nzval[:] / adj_mod.nzval[1] .+ 0.1
nlabels_fontsize = 25
edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers]
nlabels = [@sprintf("%.10e", node_labels[i]) for i in 1:nv(G)]
graphplot!(ax_middle, G, layout=layout, nlabels=nlabels, node_size=100,
    node_color=(:orange, 0.9), edge_color=edge_color, edge_width=3,
    arrow_size=25, nlabels_align=(:center, :center),
    nlabels_fontsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
hidedecorations!(ax_middle);
hidespines!(ax_middle);

display(fig)

println(PI)
##
function leicht_newmanP(A, nc::Int64)
    nc -= 1
    _, G, H, PI = leicht_newman_with_treeP(A, 0.)
    H_ind = sortperm(G[2:end],rev=true)
    H_ind = H_ind .+ 1
    pushfirst!(H_ind,1)
    H_ord = []
    if nc > length(G) 
        println("Maximum number of clusters is ", length(G)+1, ". The number of clusters has been now changed to ", length(G)+1)
        nc = length(G)
    end
    for i = 1:nc
        push!(H_ord, H[H_ind[i]][:])
    end
    ln_nc = []
    push!(ln_nc, H_ord[end][1])
    push!(ln_nc, H_ord[end][2])
    for i in eachindex(H_ord[1:end-1]), j in 1:2
        intersections = 0
        for h in eachindex(ln_nc) intersections += length(intersect(ln_nc[h], H_ord[nc-i][j])) end
            if intersections == 0
                push!(ln_nc, H_ord[nc-i][j])
            end
    end
    nc += 1                                                                
    return nc, ln_nc
end

# distances = zeros(Float64, 3, length(PI))
# for i in eachindex(PI)
#     origin = PI[i][1]-1
#     distance = 1
#     while origin != 0
#         distance += 1
#         origin = PI[origin][1]-1
#         println(origin)
#     end
#     distances[1,i] = Float64(distance)
#     distances[2,i] = PI[i][3]
#     distances[3,i] = Float64(i)
# end
function q_next(index,PI)
    i = 1
    while i <= length(PI) && PI[i][1] != PI[index][2]
        i += 1
    end
    if i > length(PI)
        return 0.
    else
        return PI[i][3]
    end
end

function add_element(PI_ind,q,PI,index)
    for i in eachindex(PI)
        if PI[i][1] == index
            push!(PI_ind, i)
            push!(q, q_next(i,PI))
        end
    end
    q_ind = sortperm(q,rev=true)
    PI_ind = PI_ind[q_ind]
    q = q[q_ind]
    return PI_ind,q
end
##

function leicht_newmanP(A, nc::Int64)
    _, G, H, PI = leicht_newman_with_treeP(A, 0.)
    if nc > length(G) 
        println("Maximum number of clusters is ", length(G)+1, ". The number of clusters has been now changed to ", length(G)+1)
        nc = length(G)
    end
    PI_ind = []
    q = []
    PI_ind,q = add_element(PI_ind,q,PI,1) 
    for j = 1:nc-2
        ind_rem = popfirst!(PI_ind)
        popfirst!(q)
        PI_ind,q = add_element(PI_ind,q,PI,PI[ind_rem][2]) 
    end
    PI_cluster = []
    for i in eachindex(PI)
        h = mod(i,2)
        if h == 0 h = 2 end
        push!(PI_cluster, [Int(floor((i+1)/2)),h])
    end
    H_ind = []
    for i in PI_ind
        push!(H_ind,PI_cluster[i])
    end
    ln_nc = []
    for i in eachindex(H_ind)
        push!(ln_nc, H[H_ind[i][1]][H_ind[i][2]])
    end                                                           
    return nc, ln_nc
end

##
f_index = 2
P = perron_frobenius(X[f_index],step=100)
nc, ln_nc = leicht_newmanP(P, 3)
ss = 0
for i in eachindex(ln_nc)
    ss += length(ln_nc[i])
end
ss

X_LN = classes_timeseries(ln_nc, X[f_index])
##
res = 1
fig = Figure()
ax = Axis3(fig[1,1])
lines!(ax, x[f_index][1, 1:res:end], x[f_index][2, 1:res:end], x[f_index][3, 1:res:end], color=X_LN[1:res:end]) 
fig