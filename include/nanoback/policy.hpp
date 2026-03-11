#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nanoback {

class PolicyEngine {
public:
    [[nodiscard]] std::vector<double> rolling_volatility(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t window
    ) const;

    [[nodiscard]] std::vector<std::int64_t> cross_sectional_rank(
        std::span<const double> values,
        std::size_t rows,
        std::size_t cols,
        bool descending
    ) const;

    [[nodiscard]] std::vector<double> minimum_variance_weights(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t window,
        double ridge,
        double leverage
    ) const;

    [[nodiscard]] std::vector<std::int64_t> momentum_targets(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t lookback,
        std::int64_t max_position
    ) const;

    [[nodiscard]] std::vector<std::int64_t> mean_reversion_targets(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t lookback,
        std::int64_t max_position
    ) const;

    [[nodiscard]] std::vector<std::int64_t> moving_average_crossover_targets(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t fast_window,
        std::size_t slow_window,
        std::int64_t max_position
    ) const;

    [[nodiscard]] std::vector<std::int64_t> volatility_filtered_momentum_targets(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t lookback,
        std::size_t vol_window,
        double volatility_ceiling,
        std::int64_t max_position
    ) const;

    [[nodiscard]] std::vector<std::int64_t> cross_sectional_momentum_targets(
        std::span<const double> close,
        std::size_t rows,
        std::size_t cols,
        std::size_t lookback,
        std::size_t winners,
        std::size_t losers,
        std::int64_t max_position
    ) const;
};

}  // namespace nanoback
