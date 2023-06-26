using MultiDimensionalClustering.AlternativeGenerator
import MarkovChainHammer.TransitionMatrix: holding_times
Q̃ = mean(BayesianGenerator(X_LN; dt = dt))
Qred = alternative_generator(Q̃, X_LN, dt, 100) 
##
step = 100
exp(Q̃ * step * dt)
exp(Qred * step * dt)
perron_frobenius(X_LN; step = step)
ht = holding_times(X_LN; dt=dt)
##
timepf = []
timeexpqt = []
timeexpqt2 = []
steps = 1:10:40000
for step in ProgressBar(steps)
    pij = perron_frobenius(X_LN; step = step)
    push!(timepf, pij)
    push!(timeexpqt, exp(Q̃ * step * dt))
    push!(timeexpqt2, exp(Qred * step * dt))
end
##
lw = 10
fig = Figure(resolution = (2000, 1500))
m = length(union(X_LN))
labelsize = 40
yaxis_names = ["P₁₁", "P₂₁", "P₃₁", "P₁₂", "P₂₂", "P₃₂", "P₁₃", "P₂₃", "P₃₃"] .* "(t)"
axis_options = (; xlabel="time", xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
axs = []
τ = τs[3] * dt
title = "Newton-Leipnik Reduced Order Matrix Entries"
ga = fig[1,1] = GridLayout()
ax = Axis(fig[1, 1]; title = title, titlesize = 50, titlegap = 100,
leftspinevisible = false,
rightspinevisible = false,
bottomspinevisible = false,
topspinevisible = false,)
hidedecorations!(ax)
for i in 1:m^2
    ii = (i-1)%m+1 
    jj = div((i-1), m) + 1
    # println(ii, jj)
    ax  = Axis(ga[ii, jj]; ylabel = yaxis_names[i], axis_options...)
    push!(axs, ax)
    pf = [timepf[j][i] for j in eachindex(timepf)]
    eqt = [timeexpqt[j][i] for j in eachindex(timepf)]
    eqt2 = [timeexpqt2[j][i] for j in eachindex(timepf)]
    GLMakie.lines!(ax, steps * dt, pf, color = (:black, 0.5), linewidth = lw, label = "Pᵢⱼ(t)")
    GLMakie.lines!(ax, steps * dt, eqt, color = (:blue, 0.5), linewidth = lw, label = "exp(Q t)")
    GLMakie.lines!(ax, steps * dt, eqt2, color = (:red, 0.5), linewidth = lw, label = "exp(Qₚ t)")
    GLMakie.ylims!(ax, (0,1))
end
axislegend(axs[1]; position = :rt, labelsize = labelsize)
display(fig)
##
save("figure/NewtonReducedModel.png", fig)