using Random, Distributions
using ProgressBars
using Plots

ENV["JULIA_DEBUG"] = "all"

const tmax = 3 # number of frames wrong
const expectedrates = [
    0.2, # delay-critical user u
    0.8, # delay-tolerant user v
]
const ζ = 3 # path loss exponent
const a = [
    0.7, # power for user u
    0.3,  # power for user v
]
const d = [
    2, # distance of user u
    1,  # distance of user v
]
const total_frames = 200000

channel_distribution = Normal(0, √2 / 2)
outdated_state_distribution = Normal(0, √2 / 2)
plot_points = ([], [])

Threads.@threads for SNR ∈ tqdm(0:1:20)
    for τ ∈ [0.862]
        successful_frames = zeros((2, 1))

        progress_bar = 1:total_frames
        # progress_bar = ProgressBar(1:total_frames)
        # set_description(progress_bar, "($SNR dB, τ = $τ)")
        for frame ∈ progress_bar
            # Step 1: Calculate g of channel for each user
            h = rand(channel_distribution, (2, 2)) * [1, 1im]
            g = h ./ sqrt.(1 .+ d .^ ζ)

            # Step 1.1: Sort channel gains
            if abs2(h[1]) > abs2(h[2])
                h[2], h[1] = h[1], h[2]
                g = h ./ sqrt.(1 .+ d .^ ζ)
            end

            # Step 1.5: Apply time-correlated small fading coefficient
            g = (
                begin
                    ωⱼ = first(rand(channel_distribution, (1, 2)) * [1, 1im])
                    h_channel = τ^tmax * h[1] + √(1 - τ^(2tmax)) * ωⱼ
                    g_channel = @. h_channel / √(1 + d[1] ^ ζ)
                end,
                begin
                    t = 1:tmax
                    ωⱼ = rand(outdated_state_distribution, (tmax, 2)) * [1, 1im]
                    h_channel = @. τ^t * h[2] + √(1 - τ^(2t)) * ωⱼ
                    g_channel = @. h_channel / √(1 + d[2] ^ ζ)
                end
            )

            #ω = rand(channel_distribution, (2, 2)) * [1, 1im]
            #h = τ^t * h + √(1 - τ^(2t)) * ω
            #g = h ./ sqrt.(1 .+ d .^ ζ)

            # Step 2: Calculate AWGN
            σ = 10^(-SNR / 20)
            n = rand(Normal(0, σ), 2)

            # Step 3: Calculate reception SNR
            # Note: We are using the actual noise value n, instead of just the SNR-based σ
            SNRuu = a[1] * abs2(g[1]) / (a[2] * abs2(g[1]) + σ^2)
            SNRvv = @. a[2] * abs2(g[2]) / σ^2
            SNRvu = @. a[1] * abs2(g[2]) / (a[2] * abs2(g[2]) + σ^2)

            success_frame = false
            for t ∈ 1:tmax
                success_transmission_vv = (1 / tmax * log2(1 + sum(SNRvv[1:t]))) >= expectedrates[2]
                success_transmission_vu = all(rate -> rate >= expectedrates[1], @. log2(1 + SNRvu[1:t]))

                if success_transmission_vu && success_transmission_vv
                    success_frame = true
                    break
                end
            end

            successful_frames += [
                log2(1 + SNRuu) >= expectedrates[1],
                success_frame
            ]
        end

        outage_rates = @. 1 - successful_frames / total_frames
        outage_rates[1] = 1

        for rate ∈ outage_rates
            if rate > 0
                push!(plot_points[1], SNR)
                push!(plot_points[2], rate)
            end
        end
    end
end

## THEORETICAL PLOTS

# Calculate theoretical values
λ = 1 ./ (1 .+ d .^ ζ)
r = 2 .^ expectedrates .- 1
φ = max(r[1] / (a[1] - a[2] * r[1]), r[2] / a[2])

SNR = LinRange(0, 45, 100)
σ = @. 10^(-SNR / 20)
ρ = @. 1 / σ^2

L = 10 # Gaver-Stehfest procedure parameter

plotly()
plot()
for τ ∈ [0.8621]
    # NOMA - HARQ-CC
    PoutTheoreticalNOMA = []
    for σ₀ ∈ σ
        FYₖ = Array{Float64}(undef, L)
        for k ∈ 1:L
            l = floor(Int, (k+1) / 2)
            l = l:min(k,L÷2)
            ŵₖ = (-1)^(L/2 + k) * sum(@. (l^(L÷2 + 1) / factorial(L÷2) * binomial(L÷2, l) * binomial(2l, l) * binomial(l, k - l)))
            t = 1:tmax
            y = 2 ^ (tmax * expectedrates[2]) - 1
            FYₖ[k] = ŵₖ / k * (
                prod(@. 1 + a[2] * λ[2] * (1 - τ^(2t)) * k * log(2) / y / σ₀^2)
                *
                (1 + (sum(@. a[2] * λ[2] * τ^(2t) * k * log(2) / (y * σ₀^2 + a[2] * λ[2] * (1 - τ^(2t)) * k * log(2)))))
            )^(-1)
        end
        push!(PoutTheoreticalNOMA, sum(FYₖ))
    end

    # OMA - Outdated CSI
    PoutTheoreticalOMA = @. [
        1 - exp(
            -2 * (2^(2 * expectedrates[1]) - 1) / ((2 - τ^(2tmax)) * ρ * λ[1]),
        ),
        1 - 2 * exp(-(2^(2 * expectedrates[2]) - 1) / ρ / λ[2]) + exp(
            -2 * (2^(2 * expectedrates[2]) - 1) / ((2 - τ^(2tmax)) * ρ * λ[2]),
        ),
    ]

    label = [
        "a"
        "b"
    ]

    plot!(
        SNR,
        PoutTheoreticalNOMA,
        yscale = :log10,
        label = "NOMA, User v, τ = $τ",
        lw = 2
    )
    plot!(
        SNR,
        PoutTheoreticalOMA,
        yscale = :log10,
        label = ["OMA, User u, τ = $τ" "OMA, User v, τ = $τ"],
        lw = 2
    )
end

plot!(
    plot_points[1],
    plot_points[2],
    seriestype = :scatter,
    yscale = :log10,
    label = "NOMA Simulation results",
)

xlabel!("SNR (dB)")
ylabel!("Outage rate")

display(plot!(legend = :best, size = (1000,800)))
