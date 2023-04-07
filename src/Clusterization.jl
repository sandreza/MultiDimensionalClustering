module Clusterization

using ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection:leicht_newman, classes_timeseries, greatest_common_cluster
using MultiDimensionalClustering.AlternativeGenerator
using MarkovChainHammer.BayesianMatrix:BayesianGenerator
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix:perron_frobenius

export cluster

struct Cluster_timeseries
    xc
    GCC
    Perr
    PerrGen
    PerrGenPert
end

function Q_performance(X_LN,Q_LN,Q_LN_pert,nc,dt,factor)
    l,_ = eigen(Q_LN_pert)
    n_tau = Int(ceil(-1/l[end-1]/dt))*factor
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

function cluster(x, t_step, nc, dt; k_clusters=1000, n_threads=12, pm_steps = 4, performance=false, factor=6, k_means=true, iteration1=1, iteration2=4)
    indices = [t_step-pm_steps:t_step+pm_steps...] 
    if k_means   
        kmn = kmeans(x, k_clusters; n_threads=n_threads, max_iters=10^6)
        X = kmn.assignments
    else
        X = x
    end
    P = perron_frobenius(X,step=t_step)
    ln_nc = leicht_newman(P,nc)
    X_LN = classes_timeseries(ln_nc, X)
    ln_nc_indices = greatest_common_cluster(X, nc, indices; progress_bar = false)
    GCC = classes_timeseries(ln_nc_indices, X)
    P_LN = perron_frobenius(X_LN,step=t_step)
    Q_LN = mean(BayesianGenerator(X_LN;dt=dt))
    Q_LN_pert = copy(Q_LN)
    for j = 1:iteration1
        Q_LN_pert = alternative_generator(Q_LN_pert, X_LN, dt,iteration2)
    end
    cluster_timeseries = Cluster_timeseries(X_LN, GCC, P_LN, Q_LN, Q_LN_pert)
    if performance
        Parr, Qarr, Qarr_pert = Q_performance(X_LN,Q_LN,Q_LN_pert,nc,dt,factor)
        return cluster_timeseries, Parr, Qarr, Qarr_pert
    else
        return cluster_timeseries
    end
end
end