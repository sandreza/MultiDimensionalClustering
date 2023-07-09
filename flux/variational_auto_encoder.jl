using Flux, GLMakie, ProgressBars
using Random
using Flux.Data, MLDatasets
Flux.Zygote.@nograd Flux.params

L1 = 48
L2 = 16

device = cpu
encoder = Chain(Dense(d, L1, leakyrelu), Parallel(tuple, Dense(L1, L2), Dense(L1, L2))) |> device
decoder = Chain(Dense(L2, L1, leakyrelu), Dense(L1, d)) |> device

using HDF5, Random, Statistics
directory = pwd() 
hfile = h5open(directory * "/data/ks_medium_res3.hdf5")
Random.seed!(12345)
x = read(hfile["u"])
close(hfile)
##
x̄ = mean(x, dims=2)
x̃ = x .- x̄
x̂ = std(x̃, dims=2)
x̃ ./= x̂ 
batch_size = 1000
dl = DataLoader(x̃[:, 1:10:end], batchsize=batch_size, shuffle=true)
d = size(x[:,1])[1]
device = cpu # where will the calculations be performed?


function reconstuct(x)
    μ, logσ = encoder(x) # decode posterior parameters
    ϵ = device(randn(Float32, size(logσ))) # sample from N(0,I)
    z = μ + ϵ .* exp.(logσ) # reparametrization
    μ, logσ, decoder(z)
end

function vae_loss(λ, x)
    len = size(x)[end]
    μ, logσ, decoder_z = reconstuct(x)

    # D_KL(q(z|x) || p(z|x))
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 - 1f0 - 2f0 * logσ)) / len # from (10) in [1]

    # E[log p(x|z)]
    logp_x_z =  -Flux.Losses.logitbinarycrossentropy(decoder_z, x, agg=sum)/len

    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))

    -logp_x_z + kl_q_p + reg
end


λ = 0 * 0.005f0
loss(x) = vae_loss(λ, x)
loss(data_sample)

η = 1e-5
opt = ADAM(η) # optimizer
ps = Flux.params(encoder, decoder) # parameters
train!(loss, ps, opt, dl, 10)

##
fig = Figure() 
ax = Axis(fig[1,1])
lines!(ax, decoder(reverse(encoder(img_sample[1])[1])))
scatter!(ax, img_sample[1], color = :red)
display(fig)

##
Flux.Zygote.@nograd sort!

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
##

##
scale = 1000
λ = 10.0f0 * scale
loss(x) = wae_loss(λ, x)
loss(data_sample)

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
    scatter!(ax, decoder(z), color = :red)
    # scatter!(ax, encoder(xval)[1])
end
display(fig)
##
xₑₙ = [encoder(x̃[:,i])[1] for i in 1:10:564705]
xₑₙ = hcat(xₑₙ...)
##
fig = Figure()
for i in 1:16
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj])
    hist!(ax, xₑₙ[i,:], bins = 100)
    xlims!(ax, -5, 5)
end
display(fig)
# hist(x[10, :], bins = 100)
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
    μ1 = mean(dist1)
    ax = Axis(fig[ii, jj]; title = string("μ = ", μ1))

    hist!(ax, dist1  , bins = 100, color = :blue)
    dist2 = x̃[1,:] .* x̃[(i-1) * 4 + 1,:]  
    μ2 = mean(dist2)
    ax2 = Axis(fig[ii, jj + 4]; title = string("μ = ", μ2))
    hist!(ax2, dist2, bins = 100, color = :red)
    xlims!(ax, -5, 5)
end
display(fig)
##
μs = mean([encoder(x[:,i])[1] for i in 1:564705])

lines(decoder(randn(L2)))
lines(decoder(μs .+ 1 * randn(16)))