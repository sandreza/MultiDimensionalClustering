using Flux, GLMakie

using Flux.Data, MLDatasets
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
x̃ = x .- mean(x, dims=2)
x̃ ./= std(x̃, dims=2)
dl = DataLoader(x̃, batchsize=batch_size, shuffle=true)
d = size(x[:,1])[1]
device = cpu # where will the calculations be performed?
L1, L2, L3 = 32, 16, 10 # layer dimensions
η = 0.01 # learning rate for ADAM optimization algorithm
batch_size = 200; # batch size for optimization
##

function train!(model_loss, model_params, opt, loader, epochs = 10)
    train_steps = 0
    "Start training for total $(epochs) epochs" |> println
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
    end
    "Total train steps: $train_steps" |> println
end

##
enc1 = Dense(d, L1, leakyrelu)
enc2 = Dense(L1, L2, leakyrelu)
dec3 = Dense(L2, L1, leakyrelu)
dec4 = Dense(L1, d)
m = Chain(enc1, enc2, dec3, dec4) |> device
##
#=
enc1 = Dense(d, L1, leakyrelu)
enc2 = Dense(L1, L2, leakyrelu)
enc3 = Dense(L2, L3, leakyrelu)
dec3 = Dense(L3, L2, leakyrelu)
dec2 = Dense(L2, L1, leakyrelu)
dec1 = Dense(L1, d)
m = Chain(enc1, enc2, enc3, dec3, dec2, dec1) |> device
=#
##

data_sample = dl |> first |> device;
img_sample = [i for i in eachcol(data_sample[:,1:7])]
loss(x) = Flux.Losses.mse(m(x), x)
loss(data_sample)

opt = ADAM(η)
ps = Flux.params(m) # parameters
train!(loss, ps, opt, dl, 5) 

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
