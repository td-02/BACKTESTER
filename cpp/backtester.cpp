#include "nanoback/backtester.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace nanoback {

namespace {

struct PendingOrder {
    std::int64_t parent_order_id{0};
    std::int64_t target_position{0};
    std::int64_t remaining_quantity{0};
    double limit_price{std::numeric_limits<double>::quiet_NaN()};
    OrderType order_type{OrderType::market};
    std::size_t ready_index{0};
    bool active{false};
};

struct RuntimeState {
    double cash{0.0};
    double turnover{0.0};
    double total_fees{0.0};
    double total_borrow_cost{0.0};
    double total_cash_yield{0.0};
    double peak_equity{0.0};
    std::int64_t submitted_orders{0};
    std::int64_t filled_orders{0};
    std::int64_t rejected_orders{0};
    std::int64_t next_parent_order_id{1};
    std::int64_t next_child_order_id{1'000'000};
    std::int64_t next_ledger_sequence{1};
    bool halted_by_risk{false};
    std::vector<std::int64_t> positions{};
    std::vector<PendingOrder> pending{};
};

[[nodiscard]] std::size_t offset(
    const std::size_t row,
    const std::size_t col,
    const std::size_t cols
) {
    return row * cols + col;
}

[[nodiscard]] double compute_fee(
    const double notional,
    const double commission_bps
) {
    return notional * (commission_bps * 1e-4);
}

[[nodiscard]] double annual_bps_to_daily_rate(
    const double annual_bps
) {
    return annual_bps * 1e-4 / 252.0;
}

[[nodiscard]] double apply_slippage(
    const double base_price,
    const std::int64_t signed_quantity,
    const double participation,
    const BacktestConfig& config
) {
    if (config.slippage_model == SlippageModel::none || signed_quantity == 0) {
        return base_price;
    }

    const auto side = signed_quantity > 0 ? 1.0 : -1.0;
    double impact_bps = config.slippage_bps;
    if (config.slippage_model == SlippageModel::volume_share) {
        impact_bps += 10'000.0 * config.volume_share_impact * participation * participation;
    }
    return base_price * (1.0 + side * impact_bps * 1e-4);
}

[[nodiscard]] double execution_base_price(
    const OrderType order_type,
    const std::int64_t signed_quantity,
    const double close_price,
    const double bid_price,
    const double ask_price,
    const bool use_bid_ask_execution,
    const double limit_price
) {
    if (order_type == OrderType::market || std::isnan(limit_price)) {
        if (use_bid_ask_execution) {
            return signed_quantity > 0 ? ask_price : bid_price;
        }
        return close_price;
    }
    if (signed_quantity > 0) {
        const auto reference = use_bid_ask_execution ? ask_price : close_price;
        return std::min(reference, limit_price);
    }
    const auto reference = use_bid_ask_execution ? bid_price : close_price;
    return std::max(reference, limit_price);
}

[[nodiscard]] bool can_fill_limit(
    const std::int64_t signed_quantity,
    const double high_price,
    const double low_price,
    const double limit_price
) {
    if (std::isnan(limit_price)) {
        return false;
    }
    if (signed_quantity > 0) {
        return low_price <= limit_price;
    }
    return high_price >= limit_price;
}

[[nodiscard]] std::int64_t clamp_target(
    const std::int64_t requested_target,
    const BacktestConfig& config,
    const std::int64_t asset_max_position
) {
    const auto max_position = asset_max_position > 0 ? asset_max_position : config.max_position;
    if (config.allow_short) {
        return std::clamp<std::int64_t>(requested_target, -max_position, max_position);
    }
    return std::clamp<std::int64_t>(requested_target, 0, max_position);
}

[[nodiscard]] double gross_exposure_after_fill(
    const std::vector<std::int64_t>& positions,
    const std::size_t target_col,
    const std::int64_t hypothetical_position,
    std::span<const double> close,
    const std::size_t row,
    const std::size_t cols,
    const std::int64_t lot_size
) {
    double gross = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
        const auto position = col == target_col ? hypothetical_position : positions[col];
        gross += std::abs(static_cast<double>(position * lot_size) * close[offset(row, col, cols)]);
    }
    return gross;
}

[[nodiscard]] double notional_limit_for_asset(
    std::span<const double> asset_notional_limits,
    const std::size_t col
) {
    if (asset_notional_limits.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const auto value = asset_notional_limits[col];
    return value > 0.0 ? value : std::numeric_limits<double>::infinity();
}

[[nodiscard]] EngineSnapshot snapshot_from_state(
    const RuntimeState& state,
    const std::size_t next_row
) {
    EngineSnapshot snapshot{};
    snapshot.next_row = next_row;
    snapshot.cash = state.cash;
    snapshot.peak_equity = state.peak_equity;
    snapshot.total_fees = state.total_fees;
    snapshot.total_borrow_cost = state.total_borrow_cost;
    snapshot.total_cash_yield = state.total_cash_yield;
    snapshot.turnover = state.turnover;
    snapshot.submitted_orders = state.submitted_orders;
    snapshot.filled_orders = state.filled_orders;
    snapshot.rejected_orders = state.rejected_orders;
    snapshot.next_parent_order_id = state.next_parent_order_id;
    snapshot.next_child_order_id = state.next_child_order_id;
    snapshot.next_ledger_sequence = state.next_ledger_sequence;
    snapshot.halted_by_risk = state.halted_by_risk;
    snapshot.positions = state.positions;
    snapshot.pending_parent_order_ids.resize(state.pending.size());
    snapshot.pending_target_positions.resize(state.pending.size());
    snapshot.pending_remaining_quantities.resize(state.pending.size());
    snapshot.pending_limit_prices.resize(state.pending.size());
    snapshot.pending_order_types.resize(state.pending.size());
    snapshot.pending_ready_indices.resize(state.pending.size());
    snapshot.pending_active.resize(state.pending.size());

    for (std::size_t idx = 0; idx < state.pending.size(); ++idx) {
        const auto& order = state.pending[idx];
        snapshot.pending_parent_order_ids[idx] = order.parent_order_id;
        snapshot.pending_target_positions[idx] = order.target_position;
        snapshot.pending_remaining_quantities[idx] = order.remaining_quantity;
        snapshot.pending_limit_prices[idx] = order.limit_price;
        snapshot.pending_order_types[idx] = static_cast<std::int8_t>(order.order_type);
        snapshot.pending_ready_indices[idx] = order.ready_index;
        snapshot.pending_active[idx] = static_cast<std::uint8_t>(order.active ? 1 : 0);
    }
    return snapshot;
}

[[nodiscard]] RuntimeState state_from_snapshot(
    const EngineSnapshot* snapshot,
    const std::size_t cols,
    const double starting_cash
) {
    RuntimeState state{};
    state.cash = snapshot ? snapshot->cash : starting_cash;
    state.turnover = snapshot ? snapshot->turnover : 0.0;
    state.total_fees = snapshot ? snapshot->total_fees : 0.0;
    state.total_borrow_cost = snapshot ? snapshot->total_borrow_cost : 0.0;
    state.total_cash_yield = snapshot ? snapshot->total_cash_yield : 0.0;
    state.peak_equity = snapshot ? snapshot->peak_equity : starting_cash;
    state.submitted_orders = snapshot ? snapshot->submitted_orders : 0;
    state.filled_orders = snapshot ? snapshot->filled_orders : 0;
    state.rejected_orders = snapshot ? snapshot->rejected_orders : 0;
    state.next_parent_order_id = snapshot ? snapshot->next_parent_order_id : 1;
    state.next_child_order_id = snapshot ? snapshot->next_child_order_id : 1'000'000;
    state.next_ledger_sequence = snapshot ? snapshot->next_ledger_sequence : 1;
    state.halted_by_risk = snapshot ? snapshot->halted_by_risk : false;
    state.positions = snapshot ? snapshot->positions : std::vector<std::int64_t>(cols, 0);
    if (state.positions.size() != cols) {
        throw std::invalid_argument("snapshot positions size must match column count");
    }
    state.pending.resize(cols);
    if (snapshot) {
        if (snapshot->pending_parent_order_ids.size() != cols ||
            snapshot->pending_target_positions.size() != cols ||
            snapshot->pending_remaining_quantities.size() != cols ||
            snapshot->pending_limit_prices.size() != cols ||
            snapshot->pending_order_types.size() != cols ||
            snapshot->pending_ready_indices.size() != cols ||
            snapshot->pending_active.size() != cols) {
            throw std::invalid_argument("snapshot pending order vectors must match column count");
        }
        for (std::size_t idx = 0; idx < cols; ++idx) {
            state.pending[idx] = PendingOrder{
                .parent_order_id = snapshot->pending_parent_order_ids[idx],
                .target_position = snapshot->pending_target_positions[idx],
                .remaining_quantity = snapshot->pending_remaining_quantities[idx],
                .limit_price = snapshot->pending_limit_prices[idx],
                .order_type = static_cast<OrderType>(snapshot->pending_order_types[idx]),
                .ready_index = snapshot->pending_ready_indices[idx],
                .active = snapshot->pending_active[idx] != 0,
            };
        }
    }
    return state;
}

[[nodiscard]] bool has_action_at(
    const CorporateAction& action,
    const std::int64_t ts,
    const std::size_t asset
) {
    return action.ex_date_timestamp == ts && action.asset == static_cast<std::int64_t>(asset);
}

[[nodiscard]] const std::vector<Venue>& effective_venues(const BacktestConfig& config) {
    if (!config.venues.empty()) {
        return config.venues;
    }
    static const std::vector<Venue> fallback{Venue{}};
    return fallback;
}

[[nodiscard]] double venue_fill_probability(const Venue& venue, double queue_ahead_fraction) {
    if (venue.fill_probability_curve.empty()) {
        return 1.0;
    }
    const auto idx = static_cast<std::size_t>(
        std::clamp<int>(
            static_cast<int>(std::floor(queue_ahead_fraction * static_cast<double>(venue.fill_probability_curve.size() - 1))),
            0,
            static_cast<int>(venue.fill_probability_curve.size() - 1)
        )
    );
    return std::clamp(venue.fill_probability_curve[idx], 0.0, 1.0);
}

[[nodiscard]] double apply_latency_penalty(
    const BacktestConfig& config,
    const double base_price,
    const std::int64_t signed_quantity,
    const double volatility_proxy,
    std::mt19937_64& rng
) {
    const auto total_latency_us = config.signal_to_order_latency_us + config.order_to_fill_latency_us;
    if (total_latency_us <= 0 && config.adverse_selection_penalty_bps == 0.0) {
        return base_price;
    }
    double latency_us = static_cast<double>(total_latency_us);
    if (config.stochastic_latency && config.latency_jitter_sigma > 0.0) {
        std::lognormal_distribution<double> ln(std::log(std::max(1.0, latency_us)), config.latency_jitter_sigma);
        latency_us = ln(rng);
    }
    const double dt_days = latency_us / (1e6 * 60.0 * 60.0 * 24.0);
    const double side = signed_quantity >= 0 ? 1.0 : -1.0;
    double out = base_price;
    if (config.latency_drift_model == BacktestConfig::LatencyDriftModel::gbm && volatility_proxy > 0.0) {
        out = base_price * std::exp(side * std::abs(volatility_proxy) * std::sqrt(std::max(dt_days, 0.0)));
    } else if (config.latency_drift_model == BacktestConfig::LatencyDriftModel::empirical && volatility_proxy > 0.0) {
        out = base_price * (1.0 + side * std::abs(volatility_proxy) * std::sqrt(std::max(dt_days, 0.0)));
    }
    if (config.adverse_velocity_threshold > 0.0 && std::abs(volatility_proxy) > config.adverse_velocity_threshold) {
        out *= (1.0 + side * config.adverse_selection_penalty_bps * 1e-4);
    }
    return out;
}

}  // namespace

BacktestResult Backtester::run(
    std::span<const std::int64_t> timestamps,
    std::span<const double> close,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> volume,
    std::span<const double> bid,
    std::span<const double> ask,
    std::span<const std::int64_t> target_positions,
    std::span<const std::int8_t> order_types,
    std::span<const double> limit_prices,
    std::span<const std::uint8_t> tradable_mask,
    std::span<const std::int64_t> asset_max_positions,
    std::span<const double> asset_notional_limits,
    const std::size_t rows,
    const std::size_t cols,
    const BacktestConfig& config,
    const EngineSnapshot* initial_snapshot,
    const std::size_t start_row,
    const std::size_t end_row
) const {
    if (rows == 0 || cols == 0) {
        BacktestResult empty{};
        empty.ending_cash = config.starting_cash;
        empty.ending_equity = config.starting_cash;
        empty.snapshot = EngineSnapshot{.cash = config.starting_cash, .peak_equity = config.starting_cash};
        return empty;
    }

    const auto matrix_size = rows * cols;
    if (timestamps.size() != rows) {
        throw std::invalid_argument("timestamps length must match row count");
    }
    if (close.size() != matrix_size ||
        high.size() != matrix_size ||
        low.size() != matrix_size ||
        volume.size() != matrix_size ||
        bid.size() != matrix_size ||
        ask.size() != matrix_size ||
        target_positions.size() != matrix_size ||
        order_types.size() != matrix_size ||
        limit_prices.size() != matrix_size) {
        throw std::invalid_argument("market and target matrices must all match rows * cols");
    }
    if (!tradable_mask.empty() && tradable_mask.size() != rows) {
        throw std::invalid_argument("tradable_mask length must match row count");
    }
    if (!asset_max_positions.empty() && asset_max_positions.size() != cols) {
        throw std::invalid_argument("asset_max_positions length must match column count");
    }
    if (!asset_notional_limits.empty() && asset_notional_limits.size() != cols) {
        throw std::invalid_argument("asset_notional_limits length must match column count");
    }
    if (config.lot_size <= 0 || config.max_position <= 0 || config.max_participation_rate <= 0.0 ||
        config.venue_volume_share_cap <= 0.0 || config.venue_volume_share_cap > 1.0 ||
        config.queue_ahead_fraction < 0.0 || config.queue_ahead_fraction >= 1.0 ||
        config.child_slice_delay_steps < 0) {
        throw std::invalid_argument("invalid sizing, participation, queue, or scheduling parameters");
    }

    const auto bounded_end_row = std::min(end_row, rows);
    if (start_row > bounded_end_row) {
        throw std::invalid_argument("start_row must be <= end_row");
    }

    BacktestResult result{};
    result.equity_curve.resize(rows);
    result.cash_curve.resize(rows);
    result.positions.resize(matrix_size);
    result.adjustment_factors.resize(matrix_size, 1.0);
    result.fills.reserve(matrix_size);
    result.audit_events.reserve(matrix_size * 3);
    result.ledger.reserve(matrix_size * 4);

    auto state = state_from_snapshot(initial_snapshot, cols, config.starting_cash);
    std::vector<double> running_adjustment(cols, 1.0);
    std::mt19937_64 rng(42);

    auto append_ledger = [&](const std::int64_t timestamp,
                             const std::int64_t order_id,
                             const std::int64_t parent_order_id,
                             const std::int64_t asset,
                             const AuditEventType type,
                             const std::int64_t quantity,
                             const std::int64_t remaining_quantity,
                             const double price,
                             const double cash_after,
                             const double equity_after,
                             const double value) {
        result.ledger.push_back(LedgerEntry{
            .sequence = state.next_ledger_sequence++,
            .timestamp = timestamp,
            .order_id = order_id,
            .parent_order_id = parent_order_id,
            .asset = asset,
            .type = type,
            .quantity = quantity,
            .remaining_quantity = remaining_quantity,
            .price = price,
            .cash_after = cash_after,
            .equity_after = equity_after,
            .value = value,
        });
    };

    if (initial_snapshot != nullptr) {
        result.audit_events.push_back(AuditEvent{
            .timestamp = timestamps[start_row],
            .order_id = 0,
            .parent_order_id = 0,
            .asset = -1,
            .type = AuditEventType::snapshot_loaded,
            .value = static_cast<double>(initial_snapshot->next_row),
        });
        append_ledger(
            timestamps[start_row], 0, 0, -1, AuditEventType::snapshot_loaded, 0, 0, 0.0, state.cash, state.peak_equity,
            static_cast<double>(initial_snapshot->next_row)
        );
    }

    for (std::size_t row = start_row; row < bounded_end_row; ++row) {
        const bool tradable = tradable_mask.empty() ? true : tradable_mask[row] != 0;
        const auto ts = timestamps[row];

        for (const auto& roll : config.future_rolls) {
            if (roll.roll_timestamp != ts) {
                continue;
            }
            if (roll.from_asset < 0 || roll.to_asset < 0 ||
                static_cast<std::size_t>(roll.from_asset) >= cols ||
                static_cast<std::size_t>(roll.to_asset) >= cols) {
                continue;
            }
            const auto from = static_cast<std::size_t>(roll.from_asset);
            const auto to = static_cast<std::size_t>(roll.to_asset);
            const auto qty = state.positions[from];
            if (qty != 0) {
                const auto from_px = close[offset(row, from, cols)];
                const auto to_px = close[offset(row, to, cols)];
                state.positions[from] = 0;
                state.positions[to] += qty;
                const auto slippage = std::abs(static_cast<double>(qty * config.lot_size) * to_px) * roll.roll_slippage_bps * 1e-4;
                state.cash -= slippage;
                state.total_fees += slippage;
                result.audit_events.push_back(AuditEvent{
                    .timestamp = ts,
                    .order_id = 0,
                    .parent_order_id = 0,
                    .asset = static_cast<std::int64_t>(to),
                    .type = AuditEventType::future_roll,
                    .value = slippage,
                });
            }
        }

        for (const auto& action : config.corporate_actions) {
            const auto asset = static_cast<std::size_t>(action.asset);
            if (asset >= cols || action.ex_date_timestamp != ts) {
                continue;
            }
            const auto idx = offset(row, asset, cols);
            const auto px = close[idx];
            if (action.action_type == CorporateActionType::split && action.ratio_or_amount > 0.0) {
                state.positions[asset] = static_cast<std::int64_t>(
                    std::llround(static_cast<double>(state.positions[asset]) * action.ratio_or_amount)
                );
                running_adjustment[asset] /= action.ratio_or_amount;
            } else if (action.action_type == CorporateActionType::dividend) {
                const auto gross = static_cast<double>(state.positions[asset] * config.lot_size) * action.ratio_or_amount;
                if (config.dividend_reinvestment && gross > 0.0 && px > 0.0) {
                    const auto add_units = static_cast<std::int64_t>(std::floor(gross / (px * config.lot_size)));
                    if (add_units > 0) {
                        state.positions[asset] += add_units;
                        state.turnover += static_cast<double>(add_units * config.lot_size) * px;
                    } else {
                        state.cash += gross;
                    }
                } else {
                    state.cash += gross;
                }
            } else if (action.action_type == CorporateActionType::spinoff) {
                state.cash += static_cast<double>(state.positions[asset] * config.lot_size) * action.ratio_or_amount;
            } else if (action.action_type == CorporateActionType::delisting) {
                const auto qty = -state.positions[asset];
                if (qty != 0) {
                    const auto notional = static_cast<double>(qty * config.lot_size) * px;
                    state.cash -= notional;
                    state.turnover += std::abs(notional);
                    state.positions[asset] = 0;
                    for (auto& order : state.pending) {
                        if (order.active && order.parent_order_id != 0) {
                            order = PendingOrder{};
                        }
                    }
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = ts,
                        .order_id = 0,
                        .parent_order_id = 0,
                        .asset = static_cast<std::int64_t>(asset),
                        .type = AuditEventType::fill_applied,
                        .value = px,
                    });
                }
            }
        }

        double short_notional = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const auto idx = offset(row, col, cols);
            const auto notional = static_cast<double>(state.positions[col] * config.lot_size) * close[idx];
            if (state.positions[col] < 0) {
                short_notional += std::abs(notional);
            }
        }
        const auto borrow_cost = short_notional * annual_bps_to_daily_rate(config.annual_borrow_bps);
        const auto cash_yield = std::max(0.0, state.cash) * annual_bps_to_daily_rate(config.annual_cash_yield_bps);
        state.cash -= borrow_cost;
        state.cash += cash_yield;
        state.total_borrow_cost += borrow_cost;
        state.total_cash_yield += cash_yield;

        for (std::size_t col = 0; col < cols; ++col) {
            if (col >= config.instruments.size()) {
                continue;
            }
            const auto& instr = config.instruments[col];
            if ((instr.type == InstrumentType::option_call || instr.type == InstrumentType::option_put) &&
                instr.expiry_timestamp == ts) {
                const auto qty = state.positions[col];
                if (qty == 0) {
                    continue;
                }
                double intrinsic = 0.0;
                const auto underlying = instr.underlying_asset >= 0 ? static_cast<std::size_t>(instr.underlying_asset) : col;
                if (underlying >= cols) {
                    continue;
                }
                const auto u_px = close[offset(row, underlying, cols)];
                if (instr.type == InstrumentType::option_call) {
                    intrinsic = std::max(0.0, u_px - instr.strike);
                } else {
                    intrinsic = std::max(0.0, instr.strike - u_px);
                }
                const auto settle = intrinsic * static_cast<double>(qty * config.lot_size);
                state.cash += settle;
                state.positions[col] = 0;
                result.audit_events.push_back(AuditEvent{
                    .timestamp = ts,
                    .order_id = 0,
                    .parent_order_id = 0,
                    .asset = static_cast<std::int64_t>(col),
                    .type = AuditEventType::option_expiry,
                    .value = settle,
                });
            }
        }

        if (!tradable && config.cancel_orders_outside_session) {
            for (auto& order : state.pending) {
                if (!order.active) {
                    continue;
                }
                result.audit_events.push_back(AuditEvent{
                    .timestamp = timestamps[row],
                    .order_id = 0,
                    .parent_order_id = order.parent_order_id,
                    .asset = -1,
                    .type = AuditEventType::order_cancelled_session,
                    .value = 0.0,
                });
                append_ledger(
                    timestamps[row], 0, order.parent_order_id, -1, AuditEventType::order_cancelled_session,
                    0, order.remaining_quantity, 0.0, state.cash, 0.0, 0.0
                );
                order = PendingOrder{};
            }
        }

        if (tradable && !state.halted_by_risk) {
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                const auto requested_target = target_positions[idx];
                const auto clamped_target = clamp_target(
                    requested_target,
                    config,
                    asset_max_positions.empty() ? config.max_position : asset_max_positions[col]
                );
                if (clamped_target != requested_target) {
                    ++state.rejected_orders;
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = 0,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::order_rejected_limit,
                        .value = static_cast<double>(requested_target),
                    });
                    append_ledger(
                        timestamps[row], 0, 0, static_cast<std::int64_t>(col), AuditEventType::order_rejected_limit,
                        requested_target, 0, 0.0, state.cash, 0.0, static_cast<double>(requested_target)
                    );
                }

                auto& existing = state.pending[col];
                if (existing.active && existing.target_position == clamped_target) {
                    continue;
                }
                if (clamped_target == state.positions[col]) {
                    if (existing.active) {
                        result.audit_events.push_back(AuditEvent{
                            .timestamp = timestamps[row],
                            .order_id = 0,
                            .parent_order_id = existing.parent_order_id,
                            .asset = static_cast<std::int64_t>(col),
                            .type = AuditEventType::order_cancelled_replace,
                            .value = 0.0,
                        });
                        append_ledger(
                            timestamps[row], 0, existing.parent_order_id, static_cast<std::int64_t>(col),
                            AuditEventType::order_cancelled_replace, 0, existing.remaining_quantity, 0.0, state.cash, 0.0, 0.0
                        );
                    }
                    existing = PendingOrder{};
                    continue;
                }

                const auto gross = gross_exposure_after_fill(
                    state.positions,
                    col,
                    clamped_target,
                    close,
                    row,
                    cols,
                    config.lot_size
                );
                const auto gross_leverage = gross / std::max(1.0, state.cash);
                const auto projected_notional = std::abs(
                    static_cast<double>(clamped_target * config.lot_size) * close[idx]
                );
                if (gross_leverage > config.max_gross_leverage ||
                    projected_notional > notional_limit_for_asset(asset_notional_limits, col)) {
                    ++state.rejected_orders;
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = 0,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::order_rejected_leverage,
                        .value = gross_leverage,
                    });
                    append_ledger(
                        timestamps[row], 0, 0, static_cast<std::int64_t>(col), AuditEventType::order_rejected_leverage,
                        clamped_target - state.positions[col], clamped_target - state.positions[col], 0.0, state.cash, 0.0, gross_leverage
                    );
                    continue;
                }

                if (existing.active) {
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = existing.parent_order_id,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::order_cancelled_replace,
                        .value = 0.0,
                    });
                    append_ledger(
                        timestamps[row], 0, existing.parent_order_id, static_cast<std::int64_t>(col),
                        AuditEventType::order_cancelled_replace, 0, existing.remaining_quantity, 0.0, state.cash, 0.0, 0.0
                    );
                }

                existing = PendingOrder{
                    .parent_order_id = state.next_parent_order_id++,
                    .target_position = clamped_target,
                    .remaining_quantity = clamped_target - state.positions[col],
                    .limit_price = limit_prices[idx],
                    .order_type = static_cast<OrderType>(order_types[idx]),
                    .ready_index = row + static_cast<std::size_t>(std::max<std::int64_t>(0, config.latency_steps)),
                    .active = true,
                };
                ++state.submitted_orders;
                result.audit_events.push_back(AuditEvent{
                    .timestamp = timestamps[row],
                    .order_id = 0,
                    .parent_order_id = existing.parent_order_id,
                    .asset = static_cast<std::int64_t>(col),
                    .type = AuditEventType::order_submitted,
                    .value = static_cast<double>(clamped_target),
                });
                append_ledger(
                    timestamps[row], 0, existing.parent_order_id, static_cast<std::int64_t>(col),
                    AuditEventType::order_submitted, existing.remaining_quantity, existing.remaining_quantity, 0.0,
                    state.cash, 0.0, static_cast<double>(clamped_target)
                );
            }
        }

        if (tradable && !state.halted_by_risk) {
            for (std::size_t col = 0; col < cols; ++col) {
                auto& order = state.pending[col];
                if (!order.active || order.ready_index > row || order.remaining_quantity == 0) {
                    continue;
                }

                const auto idx = offset(row, col, cols);
                const auto row_volume = volume[idx] > 0.0 ? volume[idx] : config.default_volume;
                if (order.order_type == OrderType::limit &&
                    !can_fill_limit(order.remaining_quantity, high[idx], low[idx], order.limit_price)) {
                    continue;
                }

                const auto venue_capacity = std::max(
                    0.0,
                    row_volume * config.venue_volume_share_cap - row_volume * config.queue_ahead_fraction
                );
                if (venue_capacity < 1.0) {
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = order.parent_order_id,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::order_waiting_queue,
                        .value = venue_capacity,
                    });
                    append_ledger(
                        timestamps[row], 0, order.parent_order_id, static_cast<std::int64_t>(col),
                        AuditEventType::order_waiting_queue, 0, order.remaining_quantity, 0.0, state.cash, 0.0, venue_capacity
                    );
                    continue;
                }

                const auto max_fill = std::max<std::int64_t>(
                    1,
                    static_cast<std::int64_t>(std::floor(std::min(
                        row_volume * config.max_participation_rate,
                        venue_capacity
                    )))
                );
                const auto desired_fill_abs = std::min<std::int64_t>(std::llabs(order.remaining_quantity), max_fill);
                const auto child_limit = config.child_order_size > 0 ? config.child_order_size : desired_fill_abs;
                const auto child_fill_abs = std::min<std::int64_t>(desired_fill_abs, child_limit);
                const auto child_fill_qty = order.remaining_quantity > 0 ? child_fill_abs : -child_fill_abs;
                const auto participation = std::min(1.0, static_cast<double>(child_fill_abs) / row_volume);
                const auto base_exec_price = apply_slippage(
                    execution_base_price(
                        order.order_type,
                        child_fill_qty,
                        close[idx],
                        bid[idx],
                        ask[idx],
                        config.use_bid_ask_execution,
                        order.limit_price
                    ),
                    child_fill_qty,
                    participation,
                    config
                );
                const auto vol_proxy = (high[idx] > 0.0) ? std::abs(high[idx] - low[idx]) / high[idx] : 0.0;
                const auto latency_exec_price = apply_latency_penalty(config, base_exec_price, child_fill_qty, vol_proxy, rng);

                const auto& venues = effective_venues(config);
                auto remaining = std::llabs(child_fill_qty);
                std::int64_t total_signed_executed = 0;
                std::int64_t total_abs_executed = 0;
                double total_notional = 0.0;
                double total_fees = 0.0;

                for (std::size_t venue_idx = 0; venue_idx < venues.size() && remaining > 0; ++venue_idx) {
                    const auto& venue = venues[venue_idx];
                    const double share = std::max(0.0, venue.volume_share);
                    std::int64_t venue_abs = venue_idx + 1 == venues.size()
                        ? remaining
                        : static_cast<std::int64_t>(std::floor(static_cast<double>(child_fill_abs) * share));
                    venue_abs = std::clamp<std::int64_t>(venue_abs, 0, remaining);
                    if (venue_abs == 0) {
                        continue;
                    }
                    const auto fill_prob = venue_fill_probability(venue, config.queue_ahead_fraction);
                    venue_abs = static_cast<std::int64_t>(std::floor(static_cast<double>(venue_abs) * fill_prob));
                    if (venue_abs <= 0) {
                        continue;
                    }
                    const auto venue_signed = child_fill_qty > 0 ? venue_abs : -venue_abs;
                    const auto venue_notional = static_cast<double>(venue_abs * config.lot_size) * latency_exec_price;
                    const auto venue_commission = compute_fee(venue_notional, config.commission_bps);
                    const auto venue_taker = venue_notional * venue.taker_fee_bps * 1e-4;
                    const auto venue_maker = venue_notional * venue.maker_fee_bps * 1e-4;
                    const auto venue_fee = venue_commission + venue_taker - venue_maker;

                    const auto venue_order_id = state.next_child_order_id++;
                    result.fills.push_back(Fill{
                        .timestamp = timestamps[row],
                        .order_id = venue_order_id,
                        .parent_order_id = order.parent_order_id,
                        .asset = static_cast<std::int64_t>(col),
                        .price = latency_exec_price,
                        .quantity = venue_signed,
                        .remaining_quantity = order.remaining_quantity - venue_signed,
                        .fee = venue_fee,
                        .venue_id = venue.venue_id,
                        .gross_price = base_exec_price,
                        .maker_fee_bps = venue.maker_fee_bps,
                        .taker_fee_bps = venue.taker_fee_bps,
                        .net_price = latency_exec_price + (venue_fee / std::max(1.0, static_cast<double>(venue_abs * config.lot_size))),
                        .order_type = order.order_type,
                    });
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = venue_order_id,
                        .parent_order_id = order.parent_order_id,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::fill_applied,
                        .value = latency_exec_price,
                    });
                    append_ledger(
                        timestamps[row], venue_order_id, order.parent_order_id, static_cast<std::int64_t>(col),
                        AuditEventType::fill_applied, venue_signed, order.remaining_quantity - venue_signed,
                        latency_exec_price, state.cash, 0.0, venue_fee
                    );

                    remaining -= venue_abs;
                    total_abs_executed += venue_abs;
                    total_signed_executed += venue_signed;
                    total_notional += venue_notional;
                    total_fees += venue_fee;
                }

                if (total_abs_executed == 0) {
                    continue;
                }
                const auto projected_cash = state.cash - static_cast<double>(total_signed_executed * config.lot_size) * latency_exec_price - total_fees;
                if (projected_cash < 0.0 && state.positions[col] >= 0 && child_fill_qty > 0) {
                    ++state.rejected_orders;
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = order.parent_order_id,
                        .asset = static_cast<std::int64_t>(col),
                        .type = AuditEventType::order_rejected_cash,
                        .value = projected_cash,
                    });
                    append_ledger(
                        timestamps[row], 0, order.parent_order_id, static_cast<std::int64_t>(col),
                        AuditEventType::order_rejected_cash, child_fill_qty, order.remaining_quantity, latency_exec_price, state.cash, 0.0, projected_cash
                    );
                    order = PendingOrder{};
                    continue;
                }

                state.cash = projected_cash;
                state.positions[col] += total_signed_executed;
                order.remaining_quantity -= total_signed_executed;
                state.turnover += total_notional;
                state.total_fees += total_fees;
                ++state.filled_orders;

                if (order.remaining_quantity == 0 || state.positions[col] == order.target_position) {
                    order = PendingOrder{};
                } else {
                    order.ready_index = row + static_cast<std::size_t>(config.child_slice_delay_steps + 1);
                }
            }
        }

        double equity = state.cash;
        for (std::size_t col = 0; col < cols; ++col) {
            const auto idx = offset(row, col, cols);
            if (config.mark_to_market) {
                equity += static_cast<double>(state.positions[col] * config.lot_size) * close[idx];
            }
            result.positions[idx] = state.positions[col];
            result.adjustment_factors[idx] = running_adjustment[col];
        }

        if (config.margin_limit > 0.0) {
            auto margin_used = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                double ratio = 1.0;
                if (col < config.instruments.size() && config.instruments[col].margin_ratio > 0.0) {
                    ratio = config.instruments[col].margin_ratio;
                }
                margin_used += std::abs(static_cast<double>(state.positions[col] * config.lot_size) * close[idx]) * ratio;
            }
            while (margin_used > config.margin_limit) {
                std::size_t victim = cols;
                double victim_notional = std::numeric_limits<double>::infinity();
                for (std::size_t col = 0; col < cols; ++col) {
                    const auto idx = offset(row, col, cols);
                    const auto notion = std::abs(static_cast<double>(state.positions[col] * config.lot_size) * close[idx]);
                    if (state.positions[col] != 0 && notion < victim_notional) {
                        victim = col;
                        victim_notional = notion;
                    }
                }
                if (victim == cols) {
                    break;
                }
                const auto idx = offset(row, victim, cols);
                const auto qty = -state.positions[victim];
                state.cash -= static_cast<double>(qty * config.lot_size) * close[idx];
                state.positions[victim] = 0;
                result.audit_events.push_back(AuditEvent{
                    .timestamp = ts,
                    .order_id = 0,
                    .parent_order_id = 0,
                    .asset = static_cast<std::int64_t>(victim),
                    .type = AuditEventType::margin_liquidation,
                    .value = victim_notional,
                });
                margin_used = 0.0;
                for (std::size_t col = 0; col < cols; ++col) {
                    const auto i2 = offset(row, col, cols);
                    double ratio = 1.0;
                    if (col < config.instruments.size() && config.instruments[col].margin_ratio > 0.0) {
                        ratio = config.instruments[col].margin_ratio;
                    }
                    margin_used += std::abs(static_cast<double>(state.positions[col] * config.lot_size) * close[i2]) * ratio;
                }
            }
        }

        result.cash_curve[row] = state.cash;
        result.equity_curve[row] = equity;
        state.peak_equity = std::max(state.peak_equity, equity);
        const auto drawdown = state.peak_equity > 0.0 ? (state.peak_equity - equity) / state.peak_equity : 0.0;
        if (drawdown > config.max_drawdown_pct && !state.halted_by_risk) {
            state.halted_by_risk = true;
            for (auto& order : state.pending) {
                order = PendingOrder{};
            }
            result.audit_events.push_back(AuditEvent{
                .timestamp = timestamps[row],
                .order_id = 0,
                .parent_order_id = 0,
                .asset = -1,
                .type = AuditEventType::risk_kill_switch,
                .value = drawdown,
            });
            append_ledger(
                timestamps[row], 0, 0, -1, AuditEventType::risk_kill_switch, 0, 0, 0.0, state.cash, equity, drawdown
            );
        }
        result.max_drawdown = std::max(result.max_drawdown, drawdown);
        if (!result.ledger.empty()) {
            result.ledger.back().equity_after = equity;
        }
    }

    result.ending_cash = state.cash;
    result.ending_equity = bounded_end_row > 0 ? result.equity_curve[bounded_end_row - 1] : state.cash;
    result.pnl = result.ending_equity - config.starting_cash;
    result.turnover = state.turnover;
    result.total_fees = state.total_fees;
    result.total_borrow_cost = state.total_borrow_cost;
    result.total_cash_yield = state.total_cash_yield;
    result.peak_equity = state.peak_equity;
    result.submitted_orders = state.submitted_orders;
    result.filled_orders = state.filled_orders;
    result.rejected_orders = state.rejected_orders;
    result.halted_by_risk = state.halted_by_risk;
    result.snapshot = snapshot_from_state(state, bounded_end_row);
    return result;
}

BacktestResult Backtester::run_ticks(
    std::span<const TickEvent> ticks,
    std::span<const std::int64_t> target_positions,
    const std::size_t cols,
    const BacktestConfig& config
) const {
    const auto rows = ticks.size();
    if (rows == 0 || cols == 0) {
        return BacktestResult{};
    }
    if (target_positions.size() != rows * cols) {
        throw std::invalid_argument("target_positions must match tick rows * cols");
    }

    std::vector<std::int64_t> timestamps(rows, 0);
    std::vector<double> close(rows * cols, 0.0);
    std::vector<double> high(rows * cols, 0.0);
    std::vector<double> low(rows * cols, 0.0);
    std::vector<double> volume(rows * cols, config.default_volume);
    std::vector<double> bid(rows * cols, 0.0);
    std::vector<double> ask(rows * cols, 0.0);
    std::vector<std::int8_t> order_types(rows * cols, static_cast<std::int8_t>(OrderType::limit));
    std::vector<double> limit_prices(rows * cols, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::uint8_t> tradable_mask(rows, 1);
    std::vector<std::int64_t> asset_max_positions(cols, config.max_position);
    std::vector<double> asset_notional_limits(cols, 0.0);
    std::vector<double> best_bid(cols, 0.0);
    std::vector<double> best_ask(cols, 0.0);

    for (std::size_t row = 0; row < rows; ++row) {
        const auto& tick = ticks[row];
        timestamps[row] = tick.timestamp_ns;
        if (tick.asset < 0 || static_cast<std::size_t>(tick.asset) >= cols) {
            continue;
        }
        const auto asset = static_cast<std::size_t>(tick.asset);
        if (tick.side == TickSide::bid) {
            best_bid[asset] = tick.price;
        } else if (tick.side == TickSide::ask) {
            best_ask[asset] = tick.price;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            const auto idx = row * cols + col;
            const auto b = best_bid[col] > 0.0 ? best_bid[col] : tick.price;
            const auto a = best_ask[col] > 0.0 ? best_ask[col] : tick.price;
            const auto m = (b > 0.0 && a > 0.0) ? (b + a) * 0.5 : tick.price;
            close[idx] = m;
            high[idx] = tick.price;
            low[idx] = tick.price;
            bid[idx] = b;
            ask[idx] = a;
            volume[idx] = std::max(1.0, tick.size);
            if (col == asset && tick.side == TickSide::trade) {
                limit_prices[idx] = tick.price;
            }
        }
    }

    return run(
        timestamps,
        close,
        high,
        low,
        volume,
        bid,
        ask,
        target_positions,
        order_types,
        limit_prices,
        tradable_mask,
        asset_max_positions,
        asset_notional_limits,
        rows,
        cols,
        config,
        nullptr,
        0,
        rows
    );
}

}  // namespace nanoback
