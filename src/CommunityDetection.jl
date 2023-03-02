module CommunityDetection 
    using LinearAlgebra
    export leicht_newman, classes_timeseries

    function modularity_matrix(A)
        N = size(A)[1]
        b = zeros(N, N)
        for i = 1:N, j = 1:N
            b[i, j] = (A[i, j] - (1 - A[i, i]) * (sum(A[j, :])) / N) / N
        end
        B = Symmetric(b + b')
        return B
    end

    function principal_vector(B::Symmetric)
        s = ones(Int, size(B)[1])
        Λ, V = eigen(B)
        v₁ = V[:, sortperm(real.(Λ))[end]]
        s[v₁.<=0] .= -1
        return s
    end

    function modularity(B, s)
        return s' * (B * s)
    end

    function modularity_eig(B)
        Λ, V = eigen(B)
        return maximum(Λ)
    end

    modularity(B::Symmetric) = modularity(B, principal_vector(B))

    function split_community(B, indices, q_min)
        Bg = B[indices, :][:, indices]
        Bg = Bg - Diagonal(sum(Bg, dims=1)[:])
        Bg = Symmetric(Bg + Bg')
        s = principal_vector(Bg)
        q = modularity(Bg)
        #q = modularity_eig(Bg)
        qq = -1.0

        if (q > q_min)
            ind1 = [i for (j, i) in enumerate(indices) if s[j] == 1]
            ind2 = [i for (j, i) in enumerate(indices) if s[j] == -1]
            qq = q
            return ind1, ind2, qq
        end
        return [], [], qq
    end

    function leicht_newman(A, q_min)
        B = modularity_matrix(A)
        n = size(A)[1]
        W, F, G = [collect(1:n)], [], []
        qOld = 0.0
        H = []
        while (length(W) > 0)
            w = popfirst!(W)
            ind1, ind2, q = split_community(B, w, q_min)
            if (length(ind1) > 0) & (length(ind2) > 0)
                W = [ind1, ind2, W...]
                push!(H, [ind1, ind2, q])
                if q > 0
                    qOld = q
                end
            else
                push!(F, w)
                push!(G, qOld)
            end
        end
        return F, G, H
    end

    function classes_timeseries(LN, X)
        L = length(X)
        ln_classes = zeros(Int64, L)
        Threads.@threads for i = 1:L
            for j in eachindex(LN)
                if X[i] in LN[j][:]
                    ln_classes[i] = j
                end
            end
        end
        return ln_classes
    end

end