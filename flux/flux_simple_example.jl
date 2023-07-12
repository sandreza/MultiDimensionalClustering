# Packages
using Flux
using GLMakie
using Statistics
using ProgressBars
X = rand(100)
Y = 0.5X + rand(100)
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y", title = "Data for linear model")
scatter!(ax, X, Y, color = :black)
display(fig)

## Preparing data in correct data structure
Xd = reduce(hcat,X)
Yd = reduce(hcat,Y)
data = [(Xd,Yd)]
## Set up Flux problem
#
# Model 
linear_model = Dense(1,1)
# Initial mapping
# Yd_0 = Tracker.data(mod(Xd))
# Setting up loss/cost function
loss(x, y) = mean((linear_model(x).-y).^2)
# Selecting parameter optimization method
opt = ADAM(0.01, (0.99, 0.999))
# Extracting parameters from model
par = Flux.params(linear_model);

nE = 1_000
for i in ProgressBar(1:nE)
    Flux.train!(loss,par,data,opt)
end
Y_model = vcat([linear_model([x]) for x in X]...)

## Plotting
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y", title = "Data for linear model")
scatter!(ax, X, Y, color = :black)
lines!(ax, X, Y_model, color = :red)
display(fig)