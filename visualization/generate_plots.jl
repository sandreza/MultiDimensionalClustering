using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using MultiDimensionalClustering.CommunityDetection
using MarkovChainHammer.BayesianMatrix
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.TransitionMatrix: perron_frobenius
using MultiDimensionalClustering, NetworkLayout, SparseArrays, Graphs, GraphMakie, Printf
Random.seed!(12345)

# create data directory if it's not there
isdir(pwd() * "/figure") ? nothing : mkdir(pwd() * "/figure")

# plot 1 
include("algorithm.jl")
save("figure/algorithm.png", fig)