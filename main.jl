using Random, Distributions
using ProgressBars

# Does this work?
ENV["JULIA_DEBUG"] = "all"

const τ = 0.6 # time correlation coefficient
const t = 3 # number of frames wrong
const expectedrates = [
    0.585, # delay-critical user u
    2 # delay-tolerant user v
]
const ζ = 2 # path loss exponent
const a = [
    0.7, # power for user u
    0.3  # power for user v
]
const d = [
    2, # distance of user u
    1  # distance of user v
]

const SNR = 10 # SNR in decibels
const total_frames = 20000;

channel_distribution = Normal(0, √2 / 2)

successful_frames = [0, 0]

frame = 0
for frame ∈ ProgressBar(1:total_frames)
    # Step 1: Calculate g of channel for each user
    h = rand(channel_distribution, (2,2)) * [1, 1im]
    g = h ./ sqrt.(1 .+ d.^ζ)

    # Step 1.1: Sort channel gains
    if abs2(h[1]) > abs2(h[2])
        h[2], h[1] = h[1], h[2]
        g = h ./ sqrt.(1 .+ d.^ζ)
    end

    # Step 1.5: Apply time-correlated small fading coefficient
    ω = rand(channel_distribution, (2,2)) * [1, 1im]
    h = τ^t * h + √(1 - τ^(2t)) * ω
    g = h ./ sqrt.(1 .+ d.^ζ)

    # Step 2: Calculate AWGN
    σ = 10^(- SNR / 20)
    n = rand(Normal(0, σ), 2)

    # Step 3: Calculate reception SNR
    # Note: We are using the actual noise value n, instead of just the SNR-based σ
    SNRuu = a[1] * abs2(g[1]) / (a[2] * abs2(g[1]) + σ^2)
    SNRvv = a[2] * abs2(g[2]) / σ^2
    SNRvu = a[1] * abs2(g[2]) / (a[2] * abs2(g[2]) + σ^2)

    receiver_rates = [
        log2(1 + SNRuu),
        log2(1 + SNRvv),
        log2(1 + SNRvu),
    ]

    global successful_frames += [
        (receiver_rates[1] >= expectedrates[1]) ? 1 : 0,
        (receiver_rates[2] >= expectedrates[2] && receiver_rates[3] >= expectedrates[1]) ? 1 : 0
    ]
end

@debug "Completed after $total_frames iterations"
@info "Packet success: $successful_frames / $total_frames"
@info "Outage rate u: $(1 - successful_frames[1] / total_frames)"
@info "Outage rate v: $(1 - successful_frames[2] / total_frames)"

# Calculate theoretical values
σ = 10^(- SNR / 20)
ρ = 1 / σ^2
λ = 1 ./ (1 .+ d.^ζ)
r = 2 .^ expectedrates .- 1
φ = max(r[1] / (a[1] - a[2] * r[1]) , r[2] / a[2])

# NOMA - Outdated CSI
PoutTheoretical = [
    1 - exp(- 2 * r[1] * σ^2 / (2 - τ^(2t)) / (a[1] - a[2] * r[1]) / λ[1]),
    1 - 2 * exp(- φ * σ^2 / λ[2]) + exp(- 2 * φ * σ^2 / (2 - τ^(2t)) / λ[2])
]

@info "Theor. Outage Rate u: $(PoutTheoretical[1])"
@info "Theor. Outage Rate v: $(PoutTheoretical[2])"
