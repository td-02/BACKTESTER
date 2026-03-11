#include "nanoback/policy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <stdexcept>
#include <vector>

namespace nanoback {

namespace {

[[nodiscard]] std::size_t offset(
    const std::size_t row,
    const std::size_t col,
    const std::size_t cols
) {
    return row * cols + col;
}

[[nodiscard]] std::int64_t signed_target(
    const double signal,
    const std::int64_t max_position
) {
    if (signal > 0.0) {
        return max_position;
    }
    if (signal < 0.0) {
        return -max_position;
    }
    return 0;
}

void validate_dimensions(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols
) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    if (close.size() != rows * cols) {
        throw std::invalid_argument("close size must match rows * cols");
    }
}

[[nodiscard]] double simple_return(
    const double previous,
    const double current
) {
    if (previous == 0.0) {
        return 0.0;
    }
    return (current / previous) - 1.0;
}

}  // namespace

std::vector<double> PolicyEngine::rolling_volatility(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t window
) const {
    validate_dimensions(close, rows, cols);
    if (window == 0) {
        throw std::invalid_argument("window must be positive");
    }

    std::vector<double> volatility(rows * cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        std::vector<double> returns(rows, 0.0);
        for (std::size_t row = 1; row < rows; ++row) {
            returns[row] = simple_return(
                close[offset(row - 1, col, cols)],
                close[offset(row, col, cols)]
            );
        }
        for (std::size_t row = window; row < rows; ++row) {
            double mean = 0.0;
            for (std::size_t idx = row - window + 1; idx <= row; ++idx) {
                mean += returns[idx];
            }
            mean /= static_cast<double>(window);

            double variance = 0.0;
            for (std::size_t idx = row - window + 1; idx <= row; ++idx) {
                const auto delta = returns[idx] - mean;
                variance += delta * delta;
            }
            volatility[offset(row, col, cols)] = std::sqrt(variance / static_cast<double>(window));
        }
    }
    return volatility;
}

std::vector<std::int64_t> PolicyEngine::cross_sectional_rank(
    const std::span<const double> values,
    const std::size_t rows,
    const std::size_t cols,
    const bool descending
) const {
    validate_dimensions(values, rows, cols);
    std::vector<std::int64_t> ranks(rows * cols, 0);
    std::vector<std::pair<double, std::size_t>> row_values(cols);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            row_values[col] = {values[offset(row, col, cols)], col};
        }
        std::sort(row_values.begin(), row_values.end(), [descending](const auto& left, const auto& right) {
            return descending ? left.first > right.first : left.first < right.first;
        });
        for (std::size_t rank = 0; rank < cols; ++rank) {
            ranks[offset(row, row_values[rank].second, cols)] = static_cast<std::int64_t>(rank + 1);
        }
    }
    return ranks;
}

std::vector<double> PolicyEngine::minimum_variance_weights(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t window,
    const double ridge,
    const double leverage
) const {
    validate_dimensions(close, rows, cols);
    if (window == 0 || ridge <= 0.0 || leverage <= 0.0) {
        throw std::invalid_argument("window, ridge, and leverage must be positive");
    }

    const auto volatility = rolling_volatility(close, rows, cols, window);
    std::vector<double> weights(rows * cols, 0.0);

    for (std::size_t row = 0; row < rows; ++row) {
        double inv_vol_sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const auto vol = volatility[offset(row, col, cols)];
            const auto weight = 1.0 / std::max(vol * vol, ridge);
            weights[offset(row, col, cols)] = weight;
            inv_vol_sum += weight;
        }
        if (inv_vol_sum == 0.0) {
            continue;
        }
        for (std::size_t col = 0; col < cols; ++col) {
            weights[offset(row, col, cols)] = leverage * weights[offset(row, col, cols)] / inv_vol_sum;
        }
    }
    return weights;
}

std::vector<std::int64_t> PolicyEngine::momentum_targets(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t lookback,
    const std::int64_t max_position
) const {
    validate_dimensions(close, rows, cols);
    if (lookback == 0 || max_position <= 0) {
        throw std::invalid_argument("lookback and max_position must be positive");
    }

    std::vector<std::int64_t> targets(rows * cols, 0);
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = lookback; row < rows; ++row) {
            const auto current = close[offset(row, col, cols)];
            const auto previous = close[offset(row - lookback, col, cols)];
            targets[offset(row, col, cols)] = signed_target(current - previous, max_position);
        }
    }
    return targets;
}

std::vector<std::int64_t> PolicyEngine::volatility_filtered_momentum_targets(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t lookback,
    const std::size_t vol_window,
    const double volatility_ceiling,
    const std::int64_t max_position
) const {
    validate_dimensions(close, rows, cols);
    if (lookback == 0 || vol_window == 0 || volatility_ceiling < 0.0 || max_position <= 0) {
        throw std::invalid_argument("lookback, vol_window, and max_position must be positive");
    }

    auto targets = momentum_targets(close, rows, cols, lookback, max_position);
    const auto volatility = rolling_volatility(close, rows, cols, vol_window);
    for (std::size_t idx = 0; idx < targets.size(); ++idx) {
        if (volatility[idx] > volatility_ceiling) {
            targets[idx] = 0;
        }
    }
    return targets;
}

std::vector<std::int64_t> PolicyEngine::cross_sectional_momentum_targets(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t lookback,
    const std::size_t winners,
    const std::size_t losers,
    const std::int64_t max_position
) const {
    validate_dimensions(close, rows, cols);
    if (lookback == 0 || winners + losers > cols || max_position <= 0) {
        throw std::invalid_argument("invalid lookback, winners/losers, or max_position");
    }

    std::vector<double> scores(rows * cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = lookback; row < rows; ++row) {
            scores[offset(row, col, cols)] = simple_return(
                close[offset(row - lookback, col, cols)],
                close[offset(row, col, cols)]
            );
        }
    }

    const auto desc_rank = cross_sectional_rank(scores, rows, cols, true);
    const auto asc_rank = cross_sectional_rank(scores, rows, cols, false);
    std::vector<std::int64_t> targets(rows * cols, 0);
    for (std::size_t idx = 0; idx < targets.size(); ++idx) {
        if (winners > 0 && desc_rank[idx] <= static_cast<std::int64_t>(winners)) {
            targets[idx] = max_position;
        } else if (losers > 0 && asc_rank[idx] <= static_cast<std::int64_t>(losers)) {
            targets[idx] = -max_position;
        }
    }
    return targets;
}

std::vector<std::int64_t> PolicyEngine::mean_reversion_targets(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t lookback,
    const std::int64_t max_position
) const {
    validate_dimensions(close, rows, cols);
    if (lookback == 0 || max_position <= 0) {
        throw std::invalid_argument("lookback and max_position must be positive");
    }

    std::vector<std::int64_t> targets(rows * cols, 0);
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = lookback; row < rows; ++row) {
            const auto current = close[offset(row, col, cols)];
            const auto previous = close[offset(row - lookback, col, cols)];
            targets[offset(row, col, cols)] = signed_target(previous - current, max_position);
        }
    }
    return targets;
}

std::vector<std::int64_t> PolicyEngine::moving_average_crossover_targets(
    const std::span<const double> close,
    const std::size_t rows,
    const std::size_t cols,
    const std::size_t fast_window,
    const std::size_t slow_window,
    const std::int64_t max_position
) const {
    validate_dimensions(close, rows, cols);
    if (fast_window == 0 || slow_window == 0 || fast_window >= slow_window || max_position <= 0) {
        throw std::invalid_argument("windows must be positive, fast_window < slow_window, and max_position positive");
    }

    std::vector<std::int64_t> targets(rows * cols, 0);
    for (std::size_t col = 0; col < cols; ++col) {
        double fast_sum = 0.0;
        double slow_sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const auto price = close[offset(row, col, cols)];
            fast_sum += price;
            slow_sum += price;

            if (row >= fast_window) {
                fast_sum -= close[offset(row - fast_window, col, cols)];
            }
            if (row >= slow_window) {
                slow_sum -= close[offset(row - slow_window, col, cols)];
            }
            if (row + 1 < slow_window) {
                continue;
            }

            const auto fast_avg = fast_sum / static_cast<double>(fast_window);
            const auto slow_avg = slow_sum / static_cast<double>(slow_window);
            targets[offset(row, col, cols)] = signed_target(fast_avg - slow_avg, max_position);
        }
    }
    return targets;
}

}  // namespace nanoback
