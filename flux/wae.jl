using Flux, GLMakie, ProgressBars
using Random
using Flux.Data, MLDatasets
Flux.Zygote.@nograd Flux.params
Flux.Zygote.@nograd sort!
using HDF5, Random, Statistics, LinearAlgebra

directory = pwd() 
hfile = h5open(directory * "/data/ks_medium_res3.hdf5")
Random.seed!(12345)
x = read(hfile["u"])
close(hfile)
d = size(x[:,1])[1]
L1 = 32 # 48
L2 = 16

activation_function = leakyrelu
device = cpu
# encoder = Chain(Dense(d, L1, activation_function), Dense(L1, L1, activation_function),  Parallel(tuple, Dense(L1, L2, activation_function), Dense(L1, L2, activation_function)), Parallel(tuple, Dense(L2, L2), Dense(L2, L2)) ) |> device
# decoder = Chain(Dense(L2, L1, activation_function), Dense(L1, L1, activation_function),  Dense(L1, d)) |> device
encoder = Chain(Dense(d, L1, activation_function),  Parallel(tuple, Dense(L1, L2), Dense(L1, L2)) ) |> device
decoder = Chain(Dense(L2, L1, activation_function),  Dense(L1, d)) |> device


##
x̄ = mean(x, dims=2)
x̃ = x .- x̄
x̂ = std(x̃, dims=2)
x̃ ./= x̂ 
batch_size = 100
dl = DataLoader(x̃[:, 1:10:end], batchsize=batch_size, shuffle=true)

normsq(x) = norm.(eachcol(x)).^2 |> device
function distance(x::AbstractMatrix)
    norm_x = normsq(x)
    norm_x .+ norm_x' - 2*x'*x
end
function distance(x::AbstractMatrix, y::AbstractMatrix)
    norm_x, norm_y = normsq(x), normsq(y)
    norm_x .+ norm_y' - 2*x'*y
end
reconstruction_loss(x, y) = mean(normsq(x - y))

function mmd_penalty(pz, qz) # with RBF kernel
    n = size(pz,2)
    n² = n*n
    n²1 = n²-n
    hlf = n²1 >> 1
    dist_pz = distance(pz)
    dist_qz = distance(qz)
    dist = distance(pz,qz)

    σₖ² = sort!(dist[:])[hlf]
    σₖ² += sort!(dist_qz[:])[hlf]    

    sum(exp.(dist_qz ./ -2σₖ²) + exp.(dist_pz ./ -2σₖ²) - 2I)/n²1 -
        2*sum(exp.(dist ./ -2σₖ²))/n²
end

function reconstuct(x)
    μ, logσ = encoder(x) # decode posterior parameters
    ϵ = device(randn(Float32, size(logσ))) # sample from N(0,I)
    z = μ + ϵ .* exp.(logσ) # reparametrization
    μ, logσ, z, decoder(z)
end

function wae_loss(λ, x)
    μ, logσ, smpl_qz, y = reconstuct(x)
    smpl_pz = device(randn(Float32, size(smpl_qz)...))
    reconstruction_loss(x, y) + λ*mmd_penalty(smpl_pz, smpl_qz)
end

function train!(model_loss, model_params, opt, loader, epochs = 10)
    train_steps = 0
    "Start training for total $(epochs) epochs" |> println
    loss_value = [0.0]
    for epoch = 1:epochs
        print("Epoch $(epoch): ")
        ℒ = 0
        for x in loader
            loss, back = Flux.pullback(model_params) do
                model_loss(x |> device)
            end
            # equivalent to tmp() = model_loss(x |> device); loss, back =  Flux.pullback(tmp(), model_params);
            grad = back(1f0)
            Flux.Optimise.update!(opt, model_params, grad)
            train_steps += 1
            ℒ += loss
        end
        println("ℒ = $ℒ")
        loss_value[1] = ℒ
    end
    "Total train steps: $train_steps" |> println
    return loss_value
end
##
scale = 1e0
λ = 10.0f0 * scale
loss(x) = wae_loss(λ, x)

η = 1e-3 / scale
opt = ADAM(η) # optimizer
ps = Flux.params(encoder, decoder) # parameters
# encoder_trained = deepcopy(encoder)
# decoder_trained = deepcopy(decoder)
train!(loss, ps, opt, dl, 1000)
##
fig = Figure()
# Random.seed!(12345)
for i in 1:9 
    ii = (i-1)÷3 + 1
    jj = (i-1)%3 + 1
    ax = Axis(fig[ii,jj])
    # xval = x[:, 10000 * i]
    xval = x̃[:, rand(1:564705)]
    lines!(ax, xval)
    μ, logσ = encoder(xval) # decode posterior parameters
    ϵ = device(randn(Float32, size(logσ))) # sample from N(0,I)
    println(logσ)
    z = μ + ϵ .* exp.(logσ) 
    scatter!(ax, decoder(z), color = (:red, 0.5))
    # scatter!(ax, encoder(xval)[1])
end
display(fig)
##
xₑₙ = [encoder(x̃[:,i])[1] for i in 1:1:564705]
xₑₙ = hcat(xₑₙ...)
##
fig = Figure()
for i in 1:16
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj])
    hist!(ax, xₑₙ[i,:], bins = 100, normalization = :pdf)
    xlims!(ax, -5, 5)
end
display(fig)
# hist(x[10, :], bins = 100)
##
shift = 10000
skip = 5
inds = 1+shift:skip:12000+shift
fig = Figure()
for i in 1:16
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj])
    lines!(ax, xₑₙ[i,inds])
    # xlims!(ax, -5, 5)
end
display(fig)

##
μ⃗ = mean(xₑₙ, dims = 2)
Σ = cov(xₑₙ, dims = 2)
normal_error =  norm(Σ - I) / norm(Σ * 0 + I)
##
xval = x[:,1]
μ, logσ = encoder(xval) # decode posterior parameters
ϵ = device(randn(Float32, size(logσ))) # sample from N(0,I)
z = μ + ϵ .* exp.(logσ) # reparametrization
μ, logσ, z, decoder(z)
##
lines(decoder(randn(L2)))
xₛₐ = [decoder(randn(L2)) for i in 1:100000]
xₛₐ = hcat(xₛₐ...)
##
fig = Figure(resolution = (2198, 1387))
N = 4
for i in 1:N^2
    ii = (i-1)÷N + 1
    jj = (i-1)%N + 1

    dist1 = xₛₐ[1,:] .* xₛₐ[(i-1) * 4 + 1,:]
    dist2 = x̃[1,:] .* x̃[(i-1) * 4 + 1,:]  
    μ1 = mean(dist1)
    μ2 = mean(dist2)
    ax = Axis(fig[ii, jj]; title = string("μgen = ", μ1, " μtrue = ", μ2))

    hist!(ax, dist1, bins = 100, color = (:blue, 0.5), normalization = :pdf)
    # ax2 = Axis(fig[ii, jj + 4]; title = string("μ = ", μ2))
    hist!(ax, dist2, bins = 100, color = (:red, 0.5), normalization = :pdf)
    xlims!(ax, -5, 5)
end
display(fig)