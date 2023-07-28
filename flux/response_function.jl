using HDF5, GLMakie, Statistics, ProgressBars

# hfile = h5open("/Users/andresouza/Desktop/response_function.h5", "r")
hfile = h5open("/Users/andresouza/Desktop/response_function_ou_more_training.h5", "r")
score = read(hfile, "score")
timeseries = read(hfile, "timeseries")
close(hfile)
##
Nt = size(timeseries)[end]
##
inds = 1:100
x₁ = timeseries[1,1,1,  inds]
x₂ = timeseries[1,2,1,  inds]
s₁ = score[1,1,1, inds]
s₂ = score[1,2,1, inds]
fig = Figure()
ax1 = Axis(fig[1,1])
# scatter!(ax1, x₁, x₂)
lines!(ax1, x₁)
ax2 = Axis(fig[1,2])
# scatter!(ax2, s₁, s₂)
lines!(ax2, s₁)
ax3 = Axis(fig[1,3])
lines!(ax3, x₁, s₁)
# lines!(ax3, x₂)
display(fig)
##
τs = 1:100 
shiftx = 16
shifty = 16
spatial_vals = zeros(32, 32)
τ = 4
for i in ProgressBar(1:32), j in 1:32
    spatial_vals[i,j]  = -mean(timeseries[i,j,1, τ:end] .* score[shiftx,shifty,1, 1:end-τ+1])
end 
##
fig = Figure() 
ax1 = Axis(fig[1,1]; title = "x response to perturbation at x = 16, y = 16", xlabel = "x", ylabel = "response")
scatter!(ax1, spatial_vals[:, 16], color = (:blue, 0.5))
lines!(ax1, spatial_vals[:, 16], color = :black)
ylims!(ax1, (-1,1))
ax2 = Axis(fig[1,2]; title = "y response to perturbation at x = 16, y = 16", xlabel = "y", ylabel = "response")
scatter!(ax2, spatial_vals[16, :], color = (:blue, 0.5))
lines!(ax2, spatial_vals[16, :], color = :black)
ylims!(ax2, (-1,1))
ax3 = Axis(fig[2, 1:2]; title = "(x,y) response to perturbation at x = 16, y = 16", xlabel = "x", ylabel = "y")
hm = heatmap!(ax3, spatial_vals, colorrange = (-1,1), colormap = :balance)
Colorbar(fig[2,3], hm)
display(fig) 
##
τs = 1:10:200
shiftx = 4
shifty = 4
numx = 4
numy = 4
x_indices = [maximum([shiftx-10, 1]), shiftx-1, shiftx, shiftx+1, minimum([shiftx+10, 32])]
y_indices = [maximum([shifty-10, 1]), shifty-1, shifty, shifty+1, minimum([shifty+10, 32])]
time_autocor = zeros(length(τs), length(x_indices), length(y_indices))
for (i,τ) in ProgressBar(enumerate(τs))
    for ii in eachindex(x_indices), jj in eachindex(y_indices)
        time_autocor[i, ii, jj]  = -mean(timeseries[x_indices[ii], y_indices[jj], 1, τ:end] .* score[shiftx, shifty, 1, 1:end-τ+1])
    end
end
##
fig = Figure() 
for ii in 1:5, jj in 1:5
    ax = Axis(fig[ii,jj]; title = "x = $(x_indices[ii]), y = $(y_indices[jj])")
    lines!(ax, τs, time_autocor[:, ii, jj]) 
    xlims!(ax, -10, 200)
    ylims!(ax, -1.0,1.0)
end
display(fig)