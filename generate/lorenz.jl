function lorenz63_data(n_clusters)
    dt = 0.01
    lor63 = Systems.lorenz()
    L = Int(1000 / dt)
    trj = Matrix(trajectory(lor63, L * dt, Δt=dt))
    res = 1
    L2 = Int(L / res)
    Dt = dt * res
    x = zeros(Float64, 3, L2)
    for i = 1:size(x)[2]-1
        x[:, i] = trj[(i-1)*res+1, :]
    end

    x = x[:, 10000:end]

    kmn = kmeans(x, n_clusters; n_threads=12, max_iters=1000000)
    X = kmn.assignments
    return x, X, Dt
end


function lorenz(x)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    return [10.0 * (x2 - x1), x1 * (28.0 - x3) - x2, x1 * x2 - 8.0 / 3.0 * x3]
end

function lorenz_data(timesteps, Δt, res, ϵ)
    x_f = zeros(3, timesteps)
    x_f[:, 1] = [14.0, 15.0, 27.0]
    step = RungeKutta4(3)
    for i in ProgressBar(2:timesteps)
        xOld = x_f[:, i-1]
        step(lorenz, xOld, Δt)
        𝒩 = randn(3)
        @inbounds @. x_f[:, i] = step.xⁿ⁺¹ + ϵ * sqrt(Δt) * 𝒩
    end
    L2 = floor(Int, timesteps / 10)
    Dt = Δt * res
    x = zeros(3, L2)
    for i in 1:L2
        @inbounds x[:, i] .= x_f[:, res*i]
    end

    return x, Dt
end

lorenz_data(; timesteps=10^7, Δt=0.005, res=1, ϵ=0.0) = lorenz_data(timesteps, Δt, res, ϵ)
x, dt = lorenz_data(timesteps=10^6)

@info "saving data for Lorenz"
hfile = h5open(pwd() * "/data/lorenz.hdf5", "w")
hfile["x"] = x
hfile["dt"] = dt
close(hfile)
@info "done saving data for Lorenz"