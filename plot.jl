using Random, Distributions
using ProgressBars
using Plots

ENV["JULIA_DEBUG"] = "all"

const t = 3 # number of frames wrong
const expectedrates = [
    0.585, # delay-critical user u
    2, # delay-tolerant user v
]
const ζ = 2 # path loss exponent
const a = [
    0.7, # power for user u
    0.3,  # power for user v
]
const d = [
    2, # distance of user u
    1,  # distance of user v
]
const total_frames = 10000

channel_distribution = Normal(0, √2 / 2)
plot_points = ([], [])

Threads.@threads for SNR ∈ tqdm(0:5:45)
    for τ ∈ [0.6, 1]
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
            ω = rand(channel_distribution, (2, 2)) * [1, 1im]
            h = τ^t * h + √(1 - τ^(2t)) * ω
            g = h ./ sqrt.(1 .+ d .^ ζ)

            # Step 2: Calculate AWGN
            σ = 10^(-SNR / 20)
            n = rand(Normal(0, σ), 2)

            # Step 3: Calculate reception SNR
            # Note: We are using the actual noise value n, instead of just the SNR-based σ
            SNRuu = a[1] * abs2(g[1]) / (a[2] * abs2(g[1]) + σ^2)
            SNRvv = a[2] * abs2(g[2]) / σ^2
            SNRvu = a[1] * abs2(g[2]) / (a[2] * abs2(g[2]) + σ^2)

            receiver_rates = [log2(1 + SNRuu), log2(1 + SNRvv), log2(1 + SNRvu)]

            successful_frames += [
                receiver_rates[1] >= expectedrates[1],
                receiver_rates[2] >= expectedrates[2] &&
                receiver_rates[3] >= expectedrates[1],
            ]
        end

        outage_rates = @. 1 - successful_frames / total_frames

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

plotly()
plot()
for τ ∈ [0.6, 1]
    # NOMA - Outdated CSI
    PoutTheoreticalNOMA = @. [
        1 - exp(-2 * r[1] * σ^2 / (2 - τ^(2t)) / (a[1] - a[2] * r[1]) / λ[1]),
        1 - 2 * exp(-φ * σ^2 / λ[2]) + exp(-2 * φ * σ^2 / (2 - τ^(2t)) / λ[2]),
    ]
    PoutTheoreticalOMA = @. [
        1 - exp(
            -2 * (2^(2 * expectedrates[1]) - 1) / ((2 - τ^(2t)) * ρ * λ[1]),
        ),
        1 - 2 * exp(-(2^(2 * expectedrates[2]) - 1) / ρ / λ[2]) + exp(
            -2 * (2^(2 * expectedrates[2]) - 1) / ((2 - τ^(2τ)) * ρ * λ[2]),
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
        label = ["NOMA, User u, τ = $τ" "NOMA, User v, τ = $τ"],
        lw = 2
    )
    plot!(
        SNR,
        PoutTheoreticalOMA,
        yscale = :log10,
        label = ["OMA, User u, τ = $τ" "NOMA, User v, τ = $τ"],
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
