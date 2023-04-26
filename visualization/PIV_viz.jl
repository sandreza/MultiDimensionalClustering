using HDF5, GLMakie, Dierckx
@info "Grabbing PIV data"
hfile = h5open("data/PIV.hdf5")
x = read(hfile["x"])
dt = read(hfile["dt"])
U = read(hfile["U"])
newf = U * x
# 63 x 79  data split into 2 fields
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/ 
@info "Starting Plot Creation"
fig = Figure(resolution=(800, 1100))
timeskip = 20
for i in 1:6
    jj = (i - 1) ÷ 3 + 1
    ii = (i - 1) % 3 + 1
    println(ii, jj)
    index = (i - 1) * timeskip + 1
    timeindex = (index-1) * dt
    titlestring = "t = $(timeindex)"
    ax = Axis(fig[ii, jj]; title=titlestring)
    u = reshape(newf[1:4977, index], (63, 79))
    v = reshape(newf[4978:end, index], (63, 79))
    spline_u = Spline2D(1:63, 1:79, u)
    spline_v = Spline2D(1:63, 1:79, v)
    stream(x, y) = Point2f(spline_v(x, y), spline_u(x, y))
    streamplot!(ax, stream, 1:63, 1:79, arrow_size=15, linewidth=2, colormap = :plasma)
    hidedecorations!(ax)
end
display(fig)
@info "Done with PIV viz"