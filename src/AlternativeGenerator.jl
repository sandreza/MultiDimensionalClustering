module AlternativeGenerator
using MarkovChainHammer.Utils: autocovariance
using ProgressBars, LsqFit, LinearAlgebra

export fit_autocorrelation, alternative_generator

model1(x, p) = @. exp(x * p[1])
model2(x, p) = @. exp(x * p[1]) * cos(p[2] * x)

function fit_autocorrelation(Λ, W, X, dt)
    λlist = ComplexF64[]
    for index in ProgressBar(2:length(Λ))
        g = W[end-index+1, :]
        λ = Λ[end-index+1]
        Nmax = ceil(Int, -1 / real(dt * λ))
        g_timeseries = [real(g[x]) for x in X]
        g_autocor = autocovariance(g_timeseries; timesteps=4 * Nmax)
        g_autocor = g_autocor / g_autocor[1]
        if abs(imag(λ)) > sqrt(eps(1.0))
            modeli = model2
        else
            modeli = model1
        end
        xdata = collect(range(0, 4 * Nmax - 1, length=length(g_autocor))) .* dt
        ydata = g_autocor
        pguess = [real(λ), imag(λ)]
        tmp = curve_fit(modeli, xdata, ydata, pguess)
        tmp.param
        push!(λlist, tmp.param[1] + tmp.param[2] * im)
    end
    Λ̃ = [reverse(λlist)..., Λ[end]]
    ll = eigenvalue_correction(Λ̃)
    return ll
end

function eigenvalue_correction(Λ)
    i = 1
    Λ̃ = copy(Λ)
    while i <= 100
        if abs(imag(Λ̃[i])) > eps(1.0)
            Λ̃[i+1] = conj(Λ̃[i])
            i += 2
        else
            i += 1
        end
    end
    return Λ̃
end

alternative_generator(Q::AbstractArray, X, dt) = alternative_generator(eigen(Q), X, dt)
function alternative_generator(E::Eigen, X, dt)
    Λ, V = E
    W = inv(V)
    Λ̃ = fit_autocorrelation(Λ, W, X, dt)
    Q̃ = real.(V * Diagonal(Λ̃) * W)
    return Q̃
end

end # module AlternativeGenerator