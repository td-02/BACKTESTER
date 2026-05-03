#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "nanoback/corp_actions.hpp"
#include "nanoback/instrument.hpp"
#include "nanoback/tick.hpp"
#include "nanoback/venue.hpp"

namespace nanoback {

enum class OrderType : std::int8_t {
    market = 0,
    limit = 1,
};

enum class SlippageModel : std::int8_t {
    none = 0,
    fixed_bps = 1,
    volume_share = 2,
};

enum class AuditEventType : std::int8_t {
    order_submitted = 0,
    order_rejected_limit = 1,
    order_rejected_leverage = 2,
    order_rejected_cash = 3,
    order_cancelled_session = 4,
    fill_applied = 5,
    risk_kill_switch = 6,
    order_waiting_queue = 7,
    order_cancelled_replace = 8,
    snapshot_loaded = 9,
    option_expiry = 10,
    future_roll = 11,
    margin_liquidation = 12,
};

struct BacktestConfig {
    enum class DataMode : std::int8_t {
        bar = 0,
        tick = 1,
    };
    enum class LatencyDriftModel : std::int8_t {
        none = 0,
        gbm = 1,
        empirical = 2,
    };

    double starting_cash{1'000'000.0};
    double commission_bps{0.0};
    double slippage_bps{0.0};
    double volume_share_impact{0.05};
    double max_participation_rate{0.25};
    double default_volume{1'000'000.0};
    std::int64_t lot_size{1};
    std::int64_t max_position{1};
    std::int64_t latency_steps{0};
    std::int64_t child_order_size{0};
    std::int64_t child_slice_delay_steps{0};
    double annual_borrow_bps{0.0};
    double annual_cash_yield_bps{0.0};
    double max_gross_leverage{10.0};
    double max_drawdown_pct{1.0};
    double queue_ahead_fraction{0.0};
    double venue_volume_share_cap{1.0};
    SlippageModel slippage_model{SlippageModel::volume_share};
    bool allow_short{true};
    bool mark_to_market{true};
    bool cancel_orders_outside_session{true};
    bool use_bid_ask_execution{false};
    bool dividend_reinvestment{false};
    DataMode data_mode{DataMode::bar};
    std::int64_t signal_to_order_latency_us{0};
    std::int64_t order_to_fill_latency_us{0};
    bool stochastic_latency{false};
    double latency_jitter_sigma{0.0};
    LatencyDriftModel latency_drift_model{LatencyDriftModel::none};
    double adverse_velocity_threshold{0.0};
    double adverse_selection_penalty_bps{0.0};
    double margin_limit{0.0};
    std::vector<Venue> venues{};
    std::vector<Instrument> instruments{};
    std::vector<FutureRoll> future_rolls{};
    std::vector<CorporateAction> corporate_actions{};
};

struct EngineSnapshot {
    std::size_t next_row{0};
    double cash{0.0};
    double peak_equity{0.0};
    double total_fees{0.0};
    double total_borrow_cost{0.0};
    double total_cash_yield{0.0};
    double turnover{0.0};
    std::int64_t submitted_orders{0};
    std::int64_t filled_orders{0};
    std::int64_t rejected_orders{0};
    std::int64_t next_parent_order_id{1};
    std::int64_t next_child_order_id{1'000'000};
    std::int64_t next_ledger_sequence{1};
    bool halted_by_risk{false};
    std::vector<std::int64_t> positions{};
    std::vector<std::int64_t> pending_parent_order_ids{};
    std::vector<std::int64_t> pending_target_positions{};
    std::vector<std::int64_t> pending_remaining_quantities{};
    std::vector<double> pending_limit_prices{};
    std::vector<std::int8_t> pending_order_types{};
    std::vector<std::size_t> pending_ready_indices{};
    std::vector<std::uint8_t> pending_active{};
};

struct Fill {
    std::int64_t timestamp{0};
    std::int64_t order_id{0};
    std::int64_t parent_order_id{0};
    std::int64_t asset{0};
    double price{0.0};
    std::int64_t quantity{0};
    std::int64_t remaining_quantity{0};
    double fee{0.0};
    std::int64_t venue_id{0};
    double gross_price{0.0};
    double maker_fee_bps{0.0};
    double taker_fee_bps{0.0};
    double net_price{0.0};
    OrderType order_type{OrderType::market};
};

struct AuditEvent {
    std::int64_t timestamp{0};
    std::int64_t order_id{0};
    std::int64_t parent_order_id{0};
    std::int64_t asset{-1};
    AuditEventType type{AuditEventType::order_submitted};
    double value{0.0};
};

struct LedgerEntry {
    std::int64_t sequence{0};
    std::int64_t timestamp{0};
    std::int64_t order_id{0};
    std::int64_t parent_order_id{0};
    std::int64_t asset{-1};
    AuditEventType type{AuditEventType::order_submitted};
    std::int64_t quantity{0};
    std::int64_t remaining_quantity{0};
    double price{0.0};
    double cash_after{0.0};
    double equity_after{0.0};
    double value{0.0};
};

struct BacktestResult {
    double ending_cash{0.0};
    double ending_equity{0.0};
    double pnl{0.0};
    double turnover{0.0};
    double total_fees{0.0};
    double total_borrow_cost{0.0};
    double total_cash_yield{0.0};
    double peak_equity{0.0};
    double max_drawdown{0.0};
    std::int64_t submitted_orders{0};
    std::int64_t filled_orders{0};
    std::int64_t rejected_orders{0};
    bool halted_by_risk{false};
    std::vector<double> equity_curve{};
    std::vector<double> cash_curve{};
    std::vector<std::int64_t> positions{};
    std::vector<double> adjustment_factors{};
    std::vector<Fill> fills{};
    std::vector<AuditEvent> audit_events{};
    std::vector<LedgerEntry> ledger{};
    EngineSnapshot snapshot{};
};

class Backtester {
public:
    [[nodiscard]] BacktestResult run(
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
        std::size_t rows,
        std::size_t cols,
        const BacktestConfig& config,
        const EngineSnapshot* initial_snapshot = nullptr,
        std::size_t start_row = 0,
        std::size_t end_row = static_cast<std::size_t>(-1)
    ) const;

    [[nodiscard]] BacktestResult run_ticks(
        std::span<const TickEvent> ticks,
        std::span<const std::int64_t> target_positions,
        std::size_t cols,
        const BacktestConfig& config
    ) const;
};

}  // namespace nanoback
