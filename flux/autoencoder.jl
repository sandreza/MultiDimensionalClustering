using Flux, GLMakie, ProgressBars
using Random
using Flux.Data, MLDatasets
Random.seed!(12345)
function get_data(batch_size)
    xtrain, _ = MLDatasets.MNIST.traindata(Float32)
    d = prod(size(xtrain)[1:2]) # input dimension
    xtrain1d = reshape(xtrain, d, :) # reshape input as a 784-dimesnonal vector (28*28)
    dl = DataLoader(xtrain1d, batchsize=batch_size, shuffle=true)
    dl, d
end

dl, d = get_data(batch_size)
##
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
dl = DataLoader(x̃, batchsize=batch_size, shuffle=true)
d = size(x[:,1])[1]
device = cpu # where will the calculations be performed?
##
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
L1, L2  = 32, 4 # layer dimensions
L1s = [16, 32, 48, 64]
L2s = collect(1:16)
search_list = []
for L1 ∈ ProgressBar(L1s)
    for L2 ∈ L2s
        η = 0.01 # learning rate for ADAM optimization algorithm
        batch_size = 1000; # batch size for optimization

        enc1 = Dense(d, L1, leakyrelu)
        enc2 = Dense(L1, L2, leakyrelu)
        dec3 = Dense(L2, L1, leakyrelu)
        dec4 = Dense(L1, d)
        m = Chain(enc1, enc2, dec3, dec4) |> device

        data_sample = dl |> first |> device;
        img_sample = [i for i in eachcol(data_sample[:,1:7])]
        loss(x) = Flux.Losses.mse(m(x), x)
        loss(data_sample)

        opt = ADAM(η)
        ps = Flux.params(m) # parameters
        lvalue = train!(loss, ps, opt, dl, 10) 
        push!(search_list, (L1, L2, lvalue[1]))
    end
end
##
losses = [lisval[3] for lisval in search_list]
##
L1 = 48
L2 = 16
η = 0.01 # learning rate for ADAM optimization algorithm
batch_size = 1000; # batch size for optimization

enc1 = Dense(d, L1, leakyrelu)
enc2 = Dense(L1, L2, leakyrelu)
dec3 = Dense(L2, L1, leakyrelu)
dec4 = Dense(L1, d)
m = Chain(enc1, enc2, dec3, dec4) |> device

data_sample = dl |> first |> device;
img_sample = [i for i in eachcol(data_sample[:,1:7])]
loss(x) = Flux.Losses.mse(m(x), x)
loss(data_sample)

opt = ADAM(η)
ps = Flux.params(m) # parameters
lvalue = train!(loss, ps, opt, dl, 10) 
##
fig = Figure()
for i in 1:9
    ii = (i-1)÷3 + 1
    jj = (i-1)%3 + 1
    ax = Axis(fig[ii,jj])
    lines!(ax, data_sample[:,i], color = :blue)
    scatter!(ax, m(data_sample[:,i]), color = :red) 
end
display(fig)
##
xnew = copy(x̃)
for i in 1:size(x, 2)
    xnew[:,i] = m(x̃[:,i])
end
##
fig = Figure()
ax  = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[1,3])
heatmap!(ax, x̃[:, 1:10000], colormap = :balance, colorrange = (-3,3))
heatmap!(ax2, xnew[:, 1:10000], colormap = :balance, colorrange = (-3,3))
heatmap!(ax3, x̃[:, 1:10000] .- xnew[:, 1:10000], colormap = :balance, colorrange = (-0.2, 0.2))

display(fig)