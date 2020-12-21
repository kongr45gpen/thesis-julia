using Random, Distributions
using ProgressBars
using Plots

ENV["JULIA_DEBUG"] = "all"

# Partial HARQ-IR
# Hybrid Automatic Repeat Request (Incremental Redundancy) for the delay-tolerant
# user.

const tmax = 3 # number of frames wrong
const expectedrates = [
    0.5, # delay-critical user u
    1.8, # delay-tolerant user v
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
const total_frames = 2000000

channel_distribution = Normal(0, √2 / 2)
outdated_state_distribution = Normal(0, √2 / 2)
plot_points = ([], [])

Threads.@threads for SNR ∈ tqdm(0:2:40)
    # for SNR in tqdm(0:1:20)
    for τ ∈ [0.862]
        successful_frames = zeros((2, 1))

        progress_bar = 1:total_frames
        # progress_bar = ProgressBar(1:total_frames)
        # set_description(progress_bar, "($SNR dB, τ = $τ)")
        for frame ∈ progress_bar
            h = rand(channel_distribution, (2, 2)) * [1, 1im]
            # Step 1.5: Apply time-correlated small fading coefficient


            h = (
                begin
                    t = repeat([tmax], 1, tmax)
                    ωⱼ = rand(channel_distribution, (tmax, 2)) * [1, 1im]
                    h_channel = @. τ^t * h[1] + √(1 - τ^(t)) * ωⱼ
                    # g_channel = @. h_channel / √(1 + d[1] ^ ζ)
                end,
                begin
                    t = 1:tmax
                    ωⱼ =
                        rand(outdated_state_distribution, (tmax, 2)) * [1, 1im]
                    h_channel = @. τ^t * h[2] + √(1 - τ^(2t)) * ωⱼ
                    # g_channel = @. h_channel / √(1 + d[2] ^ ζ)
                end,
            )

            # for t ∈ 1:tmax
            #     if abs2(h[1][t]) > abs2(h[2][t])
            #         h[2][t], h[1][t] = h[1][t], h[2][t]
            #     end
            # end

            g = (@. h[1] / √(1 + d[1]^ζ), @. h[2] / √(1 + d[2]^ζ))

            # Step 2: Calculate AWGN
            σ = 10^(-SNR / 20)
            n = rand(Normal(0, σ), 2)

            # Step 3: Calculate reception SNR
            # Note: We are using the actual noise value n, instead of just the SNR-based σ
            SNRuu = @. a[1] * abs2(g[1]) / (a[2] * abs2(g[1]) + σ^2)
            SNRvv = @. a[2] * abs2(g[2]) / σ^2
            SNRvu = @. a[1] * abs2(g[2]) / (a[2] * abs2(g[2]) + σ^2)

            success_frame = false
            # Feedback from user to base carries ACK signal if the user decodes
            # it successfully by combining the previously received signals
            for t ∈ tmax:tmax # Only assume final transmission (3 ACKs), not intermediate ones
                success_transmission_vv =
                    (1 / t * sum(@. log2(1 + SNRvv[1:t]))) >= expectedrates[2]
                # success_transmission_vu = all(rate -> rate >= expectedrates[1], @. log2(1 + SNRvu[1:t]))
                success_transmission_vu = true # Assume that u is always successfully decoded

                if success_transmission_vv && success_transmission_vu
                    success_frame = true
                    break
                end
            end

            successful_frames +=
                [log2(1 + SNRuu[tmax]) >= expectedrates[1], success_frame]
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

using SymPy

# Calculate theoretical values
λ = 1 ./ (1 .+ d .^ ζ)
r = 2 .^ expectedrates .- 1
φ = max(r[1] / (a[1] - a[2] * r[1]), r[2] / a[2])

SNR = LinRange(0, 45, 100)
σ = @. 10^(-SNR / 20)
ρ = @. 1 / σ^2

L = 10 # Gaver-Stehfest procedure parameter

FY(y, τ, σ) = begin
    @info "Called with $y $σ"
    FYₖ = Array{Float64}(undef, L)
    for k ∈ 1:L
        l = floor(Int, (k + 1) / 2)
        l = l:min(k, L ÷ 2) # typo in paper
        ŵₖ =
            (-1)^(L / 2 + k) * sum(@. (
                l^(L ÷ 2 + 1) / factorial(L ÷ 2) *
                binomial(L ÷ 2, l) *
                binomial(2l, l) *
                binomial(l, k - l)
            ))
        t = 1:tmax
        FYₖ[k] =
            ŵₖ / k *
            (
                prod(@. 1 + a[2] * λ[2] * (1 - τ^(2t)) * k * log(2) / y / σ^2) *
                (
                    1 + (sum(@. a[2] * λ[2] * τ^(2t) * k * log(2) / (
                        y * σ^2 + a[2] * λ[2] * (1 - τ^(2t)) * k * log(2)
                    )))
                )
            )^(-1)
    end
    sum(FYₖ)
end

plotlyjs()
plot()
τ = 0.8621
# for τ ∈ [0.8621]
    PoutTheoreticalNOMACC = @. FY(2^(tmax * expectedrates[2]) - 1, τ, σ)
    # PoutTheoreticalNOMAIR = @. FY(σ, τ, σ)
    PoutTheoreticalNOMAIR = @. FY((tmax * 2^(expectedrates[2]) - 1) / 1, τ, σ)
    # for σ₀ ∈ σ
    #     push!(PoutTheoreticalNOMA, )
    # end
    t = 1:tmax

    meijerg = convert(Float64, sympy.functions.special.hyper.meijerg(
        reshape([1 repeat([2], 1, tmax)], tmax + 1),
        [],
        [],
        reshape([repeat([1], 1, tmax) 0], tmax + 1),
        2^(tmax * expectedrates[2]),
    ).evalf())

    PoutUpperNOMAIR = vec(meijerg .* (1 + sum(@. τ^(2t) / (1 - τ^(2t))))^(-1) ./ prod((@. a[2] * ρ' * λ[2] * (1 - τ^(2t))), dims = 1))

    # OMA - Outdated CSI
    PoutTheoreticalOMA = @. [
        1 - exp(
            -2 * (2^(2 * expectedrates[1]) - 1) / ((2 - τ^(2tmax)) * ρ * λ[1]),
        ),
        1 - 2 * exp(-(2^(2 * expectedrates[2]) - 1) / ρ / λ[2]) + exp(
            -2 * (2^(2 * expectedrates[2]) - 1) / ((2 - τ^(2tmax)) * ρ * λ[2]),
        ),
    ]
    PoutTheoreticalNOMAInit = @. [
        1 - exp(
            -2 * r[1] * σ^2 / (2 - τ^(2tmax)) / (a[1] - a[2] * r[1]) / λ[1],
        ),
        1 - 2 * exp(-φ * σ^2 / λ[2]) +
        exp(-2 * φ * σ^2 / (2 - τ^(2tmax)) / λ[2]),
    ]

    plot!(
        SNR,
        PoutTheoreticalNOMACC,
        yscale = :log10,
        label = "NOMA-HARQ-CC, User v, τ = $τ",
        lw = 2,
    )
    plot!(
        SNR,
        PoutTheoreticalNOMAIR,
        yscale = :log10,
        label = "NOMA-HARQ-IR, User v, τ = $τ, Lower Bound",
        lw = 2,
    )
    plot!(
        SNR,
        PoutUpperNOMAIR,
        yscale = :log10,
        label = "NOMA-HARQ-IR, User v, τ = $τ, Upper Bound",
        lw = 2,
    )
    plot!(
        SNR,
        PoutTheoreticalNOMAInit,
        yscale = :log10,
        label = "Correct",
        lw = 4,
    )
    plot!(
        SNR,
        PoutTheoreticalOMA,
        yscale = :log10,
        label = ["OMA, User u, τ = $τ" "OMA, User v, τ = $τ"],
        lw = 2,
    )
# end

plot!(
    plot_points[1],
    plot_points[2],
    seriestype = :scatter,
    yscale = :log10,
    label = "NOMA Simulation results",
)

xlabel!("SNR (dB)")
ylabel!("Outage rate")

display(plot!(legend = :best, size = (1000, 800)))
