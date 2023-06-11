module GenerateFineCluster
using HDF5, GLMakie, ParallelKMeans, LinearAlgebra, Statistics, Random
using LaTeXStrings
export read_fine_cluster, plot_fine_cluster, fine_cluster

function read_fine_cluster(file)
    hfile = h5open(pwd() * "/data/" * file * "_fine_cluster.hdf5")
    X = read(hfile["X"])
    Xc = read(hfile["Xc"])
    close(hfile)
    return X, Xc
end

function plot_fine_cluster(x,X,title,figure_number; res = 1, mks = 5, azimuth=0.0pi, elevation=0.0pi)
    fig = Figure(resolution=(2000, 1600))
    colormap = :glasbey_hv_n256
    ax = Axis3(fig[1, 1], xticklabelsize=40, yticklabelsize=40, zticklabelsize=40, xlabelsize=60, ylabelsize=60, zlabelsize=60, azimuth = azimuth, elevation=elevation)
    scatter!(ax, x[1, 1:res:end], x[2, 1:res:end], x[3, 1:res:end], color=cgrad(colormap)[(X[1:res:end] .% 256 .+1)]; markersize=mks)
    Label(fig[1, 1, Top()], title; textsize=100)
    save("figure/figure" * string(figure_number[1])*string(figure_number[2]) * ".png", fig)
end

function fine_cluster(x; n_clusters = 2000, file = false, overwrite = false)
    if n_clusters > Int(round(length(x[1,:])/10))
        n_clusters = Int(round(length(x[1,:])/10))
    end
    if file != false
        if overwrite | !(isfile(pwd()*"/data/" * file * "_fine_cluster.hdf5"))
            Random.seed!(12345)
            kmn = kmeans(x, n_clusters; max_iters=10^3)
            X = kmn.assignments
            Xc = kmn.centers
            @info "Create file: " * file * "_fine_cluster.hdf5"
            hfile = h5open(pwd()*"/data/" * file * "_fine_cluster.hdf5", "w")
            hfile["X"] = X
            hfile["Xc"] = Xc
            close(hfile)
        else
            @info "File already exists. Use overwrite = true to overwrite."
        end
    end
    return nothing
end
end