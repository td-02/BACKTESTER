#include "nanoback/backtester.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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
    const BacktestConfig& config,
    const SlippageModel model
) {
    if (model == SlippageModel::none || signed_quantity == 0) {
        return base_price;
    }

    const auto side = signed_quantity > 0 ? 1.0 : -1.0;
    double impact_bps = config.slippage_bps;
    if (model == SlippageModel::volume_share) {
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
    const BacktestConfig& config
) {
    if (config.allow_short) {
        return std::clamp<std::int64_t>(requested_target, -config.max_position, config.max_position);
    }
    return std::clamp<std::int64_t>(requested_target, 0, config.max_position);
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
    const std::size_t rows,
    const std::size_t cols,
    const BacktestConfig& config
) const {
    if (rows == 0 || cols == 0) {
        return BacktestResult{
            .ending_cash = config.starting_cash,
            .ending_equity = config.starting_cash,
            .pnl = 0.0,
            .turnover = 0.0,
            .submitted_orders = 0,
            .filled_orders = 0,
            .rejected_orders = 0,
            .equity_curve = {},
            .cash_curve = {},
            .positions = {},
            .fills = {},
        };
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
    if (config.lot_size <= 0 || config.max_position <= 0 || config.max_participation_rate <= 0.0 ||
        config.venue_volume_share_cap <= 0.0 || config.venue_volume_share_cap > 1.0 ||
        config.queue_ahead_fraction < 0.0 || config.queue_ahead_fraction >= 1.0) {
        throw std::invalid_argument("invalid sizing, participation, queue, or venue cap parameters");
    }

    BacktestResult result{};
    result.equity_curve.resize(rows);
    result.cash_curve.resize(rows);
    result.positions.resize(matrix_size);
    result.fills.reserve(matrix_size);
    result.audit_events.reserve(matrix_size * 2);
    result.ledger.reserve(matrix_size * 3);

    double cash = config.starting_cash;
    double turnover = 0.0;
    double total_fees = 0.0;
    double total_borrow_cost = 0.0;
    double total_cash_yield = 0.0;
    std::vector<std::int64_t> positions(cols, 0);
    std::vector<PendingOrder> pending(cols);
    double peak_equity = config.starting_cash;
    bool halted_by_risk = false;
    std::int64_t next_parent_order_id = 1;
    std::int64_t next_child_order_id = 1'000'000;
    std::int64_t next_ledger_sequence = 1;

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
            .sequence = next_ledger_sequence++,
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

    for (std::size_t row = 0; row < rows; ++row) {
        const bool tradable = tradable_mask.empty() ? true : tradable_mask[row] != 0;

        {
            double short_notional = 0.0;
            double gross_notional = 0.0;
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                const auto notional = static_cast<double>(positions[col] * config.lot_size) * close[idx];
                gross_notional += std::abs(notional);
                if (positions[col] < 0) {
                    short_notional += std::abs(notional);
                }
            }
            const auto borrow_cost = short_notional * annual_bps_to_daily_rate(config.annual_borrow_bps);
            const auto cash_yield = std::max(0.0, cash) * annual_bps_to_daily_rate(config.annual_cash_yield_bps);
            cash -= borrow_cost;
            cash += cash_yield;
            total_borrow_cost += borrow_cost;
            total_cash_yield += cash_yield;
        }

        if (!tradable && config.cancel_orders_outside_session) {
            for (auto& order : pending) {
                if (order.active) {
                    result.audit_events.push_back(AuditEvent{
                        .timestamp = timestamps[row],
                        .order_id = 0,
                        .parent_order_id = order.parent_order_id,
                        .asset = -1,
                        .type = AuditEventType::order_cancelled_session,
                        .value = 0.0,
                    });
                    append_ledger(
                        timestamps[row],
                        0,
                        order.parent_order_id,
                        -1,
                        AuditEventType::order_cancelled_session,
                        0,
                        order.remaining_quantity,
                        0.0,
                        cash,
                        result.equity_curve[std::max<std::size_t>(0, row == 0 ? 0 : row - 1)],
                        0.0
                    );
                }
                order = PendingOrder{};
            }
        }

        if (tradable && !halted_by_risk) {
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                const auto requested_target = target_positions[idx];
                const auto clamped_target = clamp_target(requested_target, config);
                if (clamped_target != requested_target) {
                    ++result.rejected_orders;
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
                        requested_target, 0, 0.0, cash, 0.0, static_cast<double>(requested_target)
                    );
                }

                if (pending[col].active && pending[col].target_position == clamped_target) {
                    continue;
                }
                if (clamped_target == positions[col]) {
                    pending[col] = PendingOrder{};
                    continue;
                }

                const auto gross = gross_exposure_after_fill(
                    positions,
                    col,
                    clamped_target,
                    close,
                    row,
                    cols,
                    config.lot_size
                );
                const auto equity_base = std::max(1.0, cash);
                const auto gross_leverage = gross / equity_base;
                if (gross_leverage > config.max_gross_leverage) {
                    ++result.rejected_orders;
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
                        clamped_target - positions[col], clamped_target - positions[col], 0.0, cash, 0.0, gross_leverage
                    );
                    continue;
                }

                pending[col] = PendingOrder{
                    .parent_order_id = next_parent_order_id++,
                    .target_position = clamped_target,
                    .remaining_quantity = clamped_target - positions[col],
                    .limit_price = limit_prices[idx],
                    .order_type = static_cast<OrderType>(order_types[idx]),
                    .ready_index = row + static_cast<std::size_t>(std::max<std::int64_t>(0, config.latency_steps)),
                    .active = true,
                };
                ++result.submitted_orders;
                result.audit_events.push_back(AuditEvent{
                    .timestamp = timestamps[row],
                    .order_id = 0,
                    .parent_order_id = pending[col].parent_order_id,
                    .asset = static_cast<std::int64_t>(col),
                    .type = AuditEventType::order_submitted,
                    .value = static_cast<double>(clamped_target),
                });
                append_ledger(
                    timestamps[row], 0, pending[col].parent_order_id, static_cast<std::int64_t>(col),
                    AuditEventType::order_submitted, pending[col].remaining_quantity, pending[col].remaining_quantity,
                    0.0, cash, 0.0, static_cast<double>(clamped_target)
                );
            }
        }

        if (tradable && !halted_by_risk) {
            for (std::size_t col = 0; col < cols; ++col) {
                auto& order = pending[col];
                if (!order.active || order.ready_index > row || order.remaining_quantity == 0) {
                    continue;
                }

                const auto idx = offset(row, col, cols);
                const auto high_price = high[idx];
                const auto low_price = low[idx];
                const auto close_price = close[idx];
                const auto bid_price = bid[idx];
                const auto ask_price = ask[idx];
                const auto row_volume = volume[idx] > 0.0 ? volume[idx] : config.default_volume;

                if (order.order_type == OrderType::limit &&
                    !can_fill_limit(order.remaining_quantity, high_price, low_price, order.limit_price)) {
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
                        AuditEventType::order_waiting_queue, 0, order.remaining_quantity, 0.0, cash, 0.0, venue_capacity
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
                const auto fill_abs = std::min<std::int64_t>(std::llabs(order.remaining_quantity), max_fill);
                const auto fill_qty = order.remaining_quantity > 0 ? fill_abs : -fill_abs;
                const auto participation = std::min(1.0, static_cast<double>(fill_abs) / row_volume);
                const auto base_price = execution_base_price(
                    order.order_type,
                    fill_qty,
                    close_price,
                    bid_price,
                    ask_price,
                    config.use_bid_ask_execution,
                    order.limit_price
                );
                const auto exec_price = apply_slippage(
                    base_price,
                    fill_qty,
                    participation,
                    config,
                    config.slippage_model
                );
                const auto notional = static_cast<double>(fill_abs * config.lot_size) * exec_price;
                const auto fee = compute_fee(notional, config.commission_bps);
                const auto projected_cash = cash - static_cast<double>(fill_qty * config.lot_size) * exec_price - fee;
                if (projected_cash < 0.0 && positions[col] >= 0 && fill_qty > 0) {
                    ++result.rejected_orders;
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
                        AuditEventType::order_rejected_cash, fill_qty, order.remaining_quantity, exec_price, cash, 0.0, projected_cash
                    );
                    order = PendingOrder{};
                    continue;
                }

                const auto child_limit = config.child_order_size > 0 ? config.child_order_size : std::llabs(fill_qty);
                const auto child_fill_abs = std::min<std::int64_t>(std::llabs(fill_qty), child_limit);
                const auto child_fill_qty = fill_qty > 0 ? child_fill_abs : -child_fill_abs;
                const auto child_notional = static_cast<double>(child_fill_abs * config.lot_size) * exec_price;
                const auto child_fee = compute_fee(child_notional, config.commission_bps);
                const auto child_projected_cash = cash - static_cast<double>(child_fill_qty * config.lot_size) * exec_price - child_fee;
                const auto child_order_id = next_child_order_id++;

                cash = child_projected_cash;
                positions[col] += child_fill_qty;
                order.remaining_quantity -= child_fill_qty;
                turnover += child_notional;
                total_fees += child_fee;
                ++result.filled_orders;

                result.fills.push_back(Fill{
                    .timestamp = timestamps[row],
                    .order_id = child_order_id,
                    .parent_order_id = order.parent_order_id,
                    .asset = static_cast<std::int64_t>(col),
                    .price = exec_price,
                    .quantity = child_fill_qty,
                    .remaining_quantity = order.remaining_quantity,
                    .fee = child_fee,
                    .order_type = order.order_type,
                });
                result.audit_events.push_back(AuditEvent{
                    .timestamp = timestamps[row],
                    .order_id = child_order_id,
                    .parent_order_id = order.parent_order_id,
                    .asset = static_cast<std::int64_t>(col),
                    .type = AuditEventType::fill_applied,
                    .value = exec_price,
                });
                append_ledger(
                    timestamps[row], child_order_id, order.parent_order_id, static_cast<std::int64_t>(col),
                    AuditEventType::fill_applied, child_fill_qty, order.remaining_quantity, exec_price, cash, 0.0, child_fee
                );

                if (order.remaining_quantity == 0 || positions[col] == order.target_position) {
                    order = PendingOrder{};
                }
            }
        }

        double equity = cash;
        if (config.mark_to_market) {
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                equity += static_cast<double>(positions[col] * config.lot_size) * close[idx];
                result.positions[idx] = positions[col];
            }
        } else {
            for (std::size_t col = 0; col < cols; ++col) {
                const auto idx = offset(row, col, cols);
                result.positions[idx] = positions[col];
            }
        }

        result.cash_curve[row] = cash;
        result.equity_curve[row] = equity;
        peak_equity = std::max(peak_equity, equity);
        const auto drawdown = peak_equity > 0.0 ? (peak_equity - equity) / peak_equity : 0.0;
        if (drawdown > config.max_drawdown_pct && !halted_by_risk) {
            halted_by_risk = true;
            for (auto& order : pending) {
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
                timestamps[row], 0, 0, -1, AuditEventType::risk_kill_switch, 0, 0, 0.0, cash, equity, drawdown
            );
        }
        result.max_drawdown = std::max(result.max_drawdown, drawdown);
        if (!result.ledger.empty()) {
            result.ledger.back().equity_after = equity;
        }
    }

    result.ending_cash = cash;
    result.ending_equity = result.equity_curve.back();
    result.pnl = result.ending_equity - config.starting_cash;
    result.turnover = turnover;
    result.total_fees = total_fees;
    result.total_borrow_cost = total_borrow_cost;
    result.total_cash_yield = total_cash_yield;
    result.peak_equity = peak_equity;
    result.halted_by_risk = halted_by_risk;
    return result;
}

}  // namespace nanoback
