using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using Combinatorics
using MultiDimensionalClustering.CommunityDetection, ProgressBars
using MarkovChainHammer.BayesianMatrix: BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering.AlternativeGenerator
using SparseArrays, NetworkLayout, Graphs, Printf, NetworkLayout, GraphMakie, Graphs

function read_data(file)
    hfile = h5open(pwd()*"/data/" *file* ".hdf5")
    x = read(hfile["x"])
    dt = read(hfile["dt"])
    close(hfile)
    return x, dt
end

function read_fine_cluster(file)
    hfile = h5open(pwd() * "/data/" * file * "_fine_cluster.hdf5")
    X = read(hfile["X"])
    Xc = read(hfile["Xc"])
    close(hfile)
    return X, Xc
end

function read_coarse_cluster_tree(file)
    hfile = h5open(pwd()*"/data/" * file * "_coarse_cluster_tree.hdf5")
    τ = read(hfile["τ"])
    n_timescales = length(τ)
    adj_array = []
    adj_mod_array = []
    edge_numbers_array = []
    node_labels_array = []
    X_LN_array = []
    for t = 1:n_timescales
        edge_numbers = read(hfile["tree_edge_numbers_tms=($t)"])
        push!(edge_numbers_array,edge_numbers)
        N = read(hfile["tree_matrix_size_tms=($t)"])
        adj = spzeros(Int64, N, N)
        adj_mod = spzeros(Float64, N, N)
        for i in 1:edge_numbers
            ii = read(hfile["tree_i"*string(i)*"_tms=($t)"])
            jj = read(hfile["tree_j"*string(i)*"_tms=($t)"])
            modularity_value = read(hfile["tree_modularity"*string(i)*"_tms=($t)"])
            adj[ii, jj] += 1
            adj_mod[ii, jj] = modularity_value
        end
        push!(adj_array, adj)
        push!(adj_mod_array, adj_mod)
        node_labels = zeros(Float64, N)
        for i in 1:edge_numbers
            ii = read(hfile["tree_i"*string(i)*"_tms=($t)"])
            modularity_value = read(hfile["tree_modularity"*string(i)*"_tms=($t)"])
            node_labels[ii] = modularity_value
        end
        push!(node_labels_array,node_labels)
        X_LN = read(hfile["X_LN_tms=($t)"])
        push!(X_LN_array, X_LN)
    end
    close(hfile)
    return X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ
end

function plot_fine_cluster(x,X; res = 1, mks = 5)
    fig = Figure(resolution=(2000, 2000))
    ax = LScene(fig[1, 1])
    scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=X[1:res:end]; markersize=mks)
    display(fig)
end

function fine_cluster(x; n_clusters = 2000, file = false, plt = false, res = 1, mks = 5)
    if n_clusters > Int(round(length(x[1,:])/2))
        n_clusters = Int(round(length(x[1,:])/2))
    end
    Random.seed!(12345)
    kmn = kmeans(x, n_clusters; max_iters=10^6)
    X = kmn.assignments
    Xc = kmn.centers
    if file != false
        hfile = h5open(pwd()*"/data/" * file * "_fine_cluster.hdf5", "w")
        hfile["X"] = X
        hfile["Xc"] = Xc
        close(hfile)
    end
    if plt != false
        plot_fine_cluster(x,X; res = res, mks = mks)
    end 
    return X, Xc
end

function timescales(λ)
    λ = real.(λ)
    ind = sortperm(λ,rev=true)
    λ = λ[ind]
    τ = Float64[]
    push!(τ, -1/λ[2])
    λ_len = length(λ)
    for i in 3:λ_len
        if (λ[i-1] - λ[i]) > 0.0001 push!(τ, -1/λ[i]) end
    end
    return τ
end

function plot_coarse_cluster_tree(x,X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ; res = 5, mks = 10)
    fig = Figure(resolution=(5000, 2000))
    layout = Buchheim()
    set_theme!(backgroundcolor=:white)
    n_timescales = length(τ)
    println(n_timescales)
    for t = 1:n_timescales
        ax1 = LScene(fig[1,t])
        scatter!(ax1, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=X_LN_array[t][1:res:end]; markersize=mks)
        ax2 = Axis(fig[2,t]; title="t=$(round(τ[t],sigdigits=3))", titlesize=30)
        G = SimpleDiGraph(adj_array[t])
        transparancy = 0.4 * adj_mod_array[t].nzval[:] / adj_mod_array[t].nzval[1] .+ 0.1
        nlabels_fontsize = 25
        edge_color = [(:red, transparancy[i]) for i in 1:edge_numbers_array[t]]
        nlabels = [@sprintf("%.0e", node_labels_array[t][i]) for i in 1:nv(G)]
        graphplot!(ax2, G, layout=layout, nlabels=nlabels, node_size=100,
            node_color=(:orange, 0.9), edge_color=edge_color, edge_width=3,
            arrow_size=25, nlabels_align=(:center, :center),
            nlabels_fontsize=nlabels_fontsize, tangents=((0, -1), (0, -1)))
        hidedecorations!(ax2)
        hidespines!(ax2);
    end
    display(fig)
end

function coarse_cluster_tree(x, X, dt, n_timescales; file=false, plt=true, res = 5, mks = 10)
    Q = mean(BayesianGenerator(X; dt=dt))
    Λ, _ = eigen(Q)
    τ = timescales(Λ)
    t_steps = Int.(round.(τ ./dt)) 
    q_min = 1e-16   
    adj_array = []
    adj_mod_array = []
    edge_numbers_array = []
    node_labels_array = []
    X_LN_array = []
    if file != false
        filename = "/data/" * file * "_coarse_cluster_tree.hdf5"
        hfile = h5open(pwd()*filename, "w")
        hfile["τ"] = τ[1:n_timescales]
    end
    for t in 1:n_timescales
        P = perron_frobenius(X,step=t_steps[t])
        F, _, _, PI = leicht_newman_with_tree(P, q_min)
        edge_numbers = length(PI)
        push!(edge_numbers_array,edge_numbers)
        N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
        adj = spzeros(Int64, N, N)
        adj_mod = spzeros(Float64, N, N)
        for i in 1:edge_numbers
            ii = PI[i][1]
            jj = PI[i][2]
            modularity_value = PI[i][3]
            adj[ii, jj] += 1
            adj_mod[ii, jj] = modularity_value
        end
        push!(adj_array, adj)
        push!(adj_mod_array, adj_mod)
        node_labels = zeros(Float64, N)
        for i in 1:edge_numbers
            ii = PI[i][1]
            modularity_value = PI[i][3]
            node_labels[ii] = modularity_value
        end
        push!(node_labels_array,node_labels)
        X_LN = classes_timeseries(F, X)
        push!(X_LN_array, X_LN)
        if file != false
            hfile["X_LN_tms=($t)"] = X_LN
            hfile["tree_edge_numbers_tms=($t)"] = length(PI)
            hfile["Flength_tms=($t)"] = length(F)
            hfile["tree_matrix_size_tms=($t)"] = N
            for i in eachindex(PI)
                hfile["tree_i"*string(i)*"_tms=($t)"] = PI[i][1]
                hfile["tree_j"*string(i)*"_tms=($t)"] = PI[i][2]
                hfile["tree_modularity"*string(i)*"_tms=($t)"] = PI[i][3]
            end
        end
    end
    if file != false
        close(hfile)
    end
    if plt != false
        plot_coarse_cluster_tree(x,X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ[1:n_timescales]; res = 5, mks = 10)
    end
    return X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ[1:n_timescales]
end

function Q_performance(X_LN,Q_LN,Q_LN_pert,dt,factor)
    l,_ = eigen(Q_LN_pert)
    n_tau = Int(ceil(-1/real(l[end-1])/dt))*factor
    nc = size(Q_LN_pert)[1]
    Parr = zeros(Float64,n_tau,nc,nc)
    Qarr = zeros(Float64, n_tau,nc,nc)
    Qarr_pert = zeros(Float64,n_tau,nc,nc)   
    for i = 0:n_tau-1
        Parr[i+1,:,:] = perron_frobenius(X_LN,step=i+1)
        Qarr[i+1,:,:] = exp(Q_LN * dt* i)
        Qarr_pert[i+1,:,:] = exp(Q_LN_pert * dt* i)
    end
    return Parr, Qarr, Qarr_pert
end

function coarse_cluster(x, X, dt, indices, τ, nc; factor=10, iteration1=3, iteration2=5, file=false)
    indices_len = length(indices)
    X_len = length(X)
    t_steps = zeros(Int64, 3, indices_len)
    for i in eachindex(indices)
        t_steps[1,i] = Int(round(τ[i] *0.9/dt))
        t_steps[2,i] = Int(round(τ[i] /dt))
        t_steps[3,i] = Int(round(τ[i] *1.1/dt))
    end
    X_LN = zeros(Int64, X_len, 3)
    nc_output = zeros(Int64, 3, indices_len)
    X_LN_array = []
    Q_array = []
    Q_pert_array = []
    Pt_array = []
    Qt_array = []
    Qt_pert_array = []
    if file != false
        clustername = "/data/" * file * "_coarse_cluster.hdf5"
        hfile = h5open(pwd() * clustername, "w")
    end
    for i = 1:indices_len
        for j = 1:3
            P = perron_frobenius(X,step=t_steps[j,i])
            nc_out, ln_nc = leicht_newman(P,nc[i])
            nc_output[j,i] = nc_out
            X_LN[:,j] = classes_timeseries(ln_nc, X)
        end
        if nc[i] <= 8
            score1, X_LN[:,1] = label_ordering(X_LN[:,2], X_LN[:,1])
            score2, X_LN[:,3] = label_ordering(X_LN[:,2], X_LN[:,3])
            score = (score1 + score2) / 2
        end
        Q_LN = mean(BayesianGenerator(X_LN[:,2];dt=dt))
        Q_LN_pert = copy(Q_LN)
        for _ in 1:iteration1
            Q_LN_pert = alternative_generator(Q_LN_pert,X_LN[:,2],dt,iteration2)
        end
        push!(Q_array, Q_LN)
        push!(Q_pert_array, Q_LN_pert)
        Pt, Qt, Qt_pert = Q_performance(X_LN[:,2],Q_LN,Q_LN_pert,dt,factor)
        push!(Pt_array, Pt)
        push!(Qt_array, Qt)
        push!(Qt_pert_array, Qt_pert)
        push!(X_LN_array, X_LN)
        if file != true
            hfile["X_LN_tms=($i])"] = X_LN
            hfile["Q_tms=($i)"] = Q_LN
            hfile["Q_pert_tms=($i)"] = Q_LN_pert
            hfile["Pt_array_tms=($i)"] = Pt
            hfile["Qt_array_tms=($i)"] = Qt
            hfile["Qt_pert_tms=($i)"] = Qt_pert
            hfile["score_tms=($i)"] = score
        end
    end
    if file != false
        hfile["t_steps"] = t_steps
        close(hfile)
    end
    return nc_output, X_LN_array, Q_array, Q_pert_array, Pt_array, Qt_array, Qt_pert_array
end

function label_ordering(cc_minus, cc_plus)
    cc_plus_len = length(union(cc_plus))
    perms_start = [1:cc_plus_len...]
    perms_array = []
    perms = permutations(perms_start)
    for p in perms
        push!(perms_array,p)
    end
    cc_plus_temp = copy(cc_plus)
    cc_plus_ord = copy(cc_plus)
    ScoreOld = 1.
    ScoreNew = 1.
    for i in eachindex(perms_array)
        for j in eachindex(cc_plus)
            cc_plus_temp[j] = findall(y->y==cc_plus[j],perms_array[i])[1]
        end
        ScoreNew = sum(diff_check.(cc_minus, cc_plus_temp))/length(cc_minus)
        if ScoreNew < ScoreOld
            cc_plus_ord = copy(cc_plus_temp)
            ScoreOld = ScoreNew
        end
    end
    return ScoreOld, cc_plus_ord
end

function diff_check(a,b)
    if a == b 
        return 0.
    else 
        return 1.
    end
end

function plot_coarse_cluster1(x, dt, t_steps, X_LN_array, Q_array; res = 1, mks = 5)
    indices_len = size(t_steps)[2]
    g_Qarr = []
    kwargs_edges = []
    kwargs_nodes = []
    kwargs_arrows = []
    colormap = :glasbey_hv_n256
    for ii = 1:indices_len
        Q_graph = Q_array[ii]
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
        elabels_fontsize = 180
        nlabels_fontsize = 180
        node_size = 100.0
        edge_width_Q = [5.0 for i in 1:ne(g_Q)]
        arrow_size_Q = [20.0 for i in 1:ne(g_Q)]
        node_labels_Q = repr.(1:nv(g_Q))
        push!(kwargs_edges, (; elabels=elabels, elabels_color=elabels_color, elabels_fontsize=elabels_fontsize, edge_color=edge_color_Q, edge_width=edge_width_Q))
        push!(kwargs_nodes, (; node_color=node_color, node_size=node_size, nlabels=node_labels_Q, nlabels_fontsize=nlabels_fontsize))
        push!(kwargs_arrows, (; arrow_size=arrow_size_Q))
        push!(g_Qarr, g_Q)
    end
    set_theme!(backgroundcolor=:white)
    fig = Figure(resolution=(2000, 4000))
    for i = 1:indices_len
        if i <= Int(indices_len/2)
            ax = Axis(fig[1, i]; title="t=$(round(t_steps[2,i]*dt,sigdigits=4))", titlesize=30)
        else
            ax = Axis(fig[3, i-Int(indices_len/2)]; title="t=$(round(t_steps[2,i]*dt,sigdigits=4))", titlesize=30)
        end
        hidedecorations!(ax); hidespines!(ax)
        graphplot!(ax, g_Qarr[i]; kwargs_edges[i]..., kwargs_nodes[i]..., kwargs_arrows[i]...)
    end
    for i = 1:indices_len
        if i <= Int(indices_len/2)
            ax = LScene(fig[2, i])
        else
            ax = LScene(fig[4, i-Int(indices_len/2)])
        end
        scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[i][1:res:end,1]]; markersize=mks)
        scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[X_LN_array[i][1:res:end,3]]; markersize=mks)
    end
    display(fig)
end

function plot_coarse_cluster2(dt, t_steps, Pt_array, Qt_array, Qt_pert_array, index; res = 1, mks = 5)
    PFdim = length(Pt_array[index][1,1,:])
    PFlen = length(Pt_array[index][:,1,1])
    xax = [dt:dt:PFlen*dt...]
    set_theme!(backgroundcolor=:white)
    fig = Figure(resolution=(2000, 2000))
    for i in 1:PFdim
        for j in 1:PFdim
            ax = Axis(fig[j, i])
            lines!(ax,xax, Pt_array[index][:,i,j],color=:red)
            lines!(ax,xax, Qt_array[index][:,i,j],color=:black)
            lines!(ax,xax, Qt_pert_array[index][:,i,j],color=:blue)
        end
    end
    display(fig)
end
##
files = ["potential_well", "lorenz", "newton", "kuramoto", "PIV"]
file_index = 2
x, dt = read_data(files[file_index])
##
######## DANGER #########
X,Xc = fine_cluster(x; n_clusters = 200, file = files[file_index], plt = true, res = 1, mks = 5);
##
X, Xc = read_fine_cluster(files[file_index])
plot_fine_cluster(x,X)
##
######## DANGER #########
X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ = coarse_cluster_tree(x,X,dt,8; file=files[file_index], plt=true)
##
X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ = read_coarse_cluster_tree(files[file_index]);
plot_coarse_cluster_tree(x,X_LN_array, adj_array, adj_mod_array, node_labels_array, edge_numbers_array, τ; res = 5, mks = 5)
##
######## DANGER #########
nc = [3,3,3,3,3,3,3,3]
indices = [1,2,3,4,5,6,7,8]
coarse_cluster(x, X, dt, indices, τ, nc; factor=10, iteration1=3, iteration2=5, file=files[file_index]);
##
file = files[file_index]
# function read_coarse_cluster(file)
hfile = h5open(pwd()*"/data/" * file * "_coarse_cluster.hdf5")
t_steps = read(hfile["t_steps"])
indices_len = size(t_steps)[2]
X_LN_array = []
Q_array = []
Q_pert_array = []
Pt_array = []
Qt_array = []
Qt_pert_array = []
score = []
for i = 1:indices_len
    push!(X_LN_array, read(hfile["X_LN_tms=($i])"]))
    push!(Q_array, read(hfile["Q_tms=($i)"]))
    push!(Q_pert_array, read(hfile["Q_pert_tms=($i)"]))
    push!(Pt_array, read(hfile["Pt_array_tms=($i)"]))
    push!(Qt_array, read(hfile["Qt_array_tms=($i)"]))
    push!(Qt_pert_array, read(hfile["Qt_pert_tms=($i)"]))
    push!(score, read(hfile["score_tms=($i)"]))
end
close(hfile)
##
plot_coarse_cluster1(x, dt, t_steps, X_LN_array, Q_array)


plot_coarse_cluster2(dt, t_steps, Pt_array, Qt_array, Qt_pert_array, 3)

t_steps[2,4]

t_steps[2,:]

τ

Int.(round.(τ ./dt))

##
tmp = kmeans(x, 16; max_iters=10^3)
tmp.assignments
tmp.centers