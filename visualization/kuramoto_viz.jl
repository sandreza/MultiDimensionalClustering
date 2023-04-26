using HDF5, GLMakie
directory = pwd() 
hfile = h5open(directory * "/data/ks_medium_res3.hdf5")

x = read(hfile["u"])
##
fig = Figure(resolution=(300, 1300))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="t")
skip = 20
indices = 1:skip:60000
ts = (indices .- 1) .* skip
xs = collect(1:64)
heatmap!(ax, x[:, indices], colormap=:balance, colorrange=(-2.5, 2.5), interpolate=true)
display(fig)