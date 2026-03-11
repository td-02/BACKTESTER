#include "nanoback/backtester.hpp"
#include "nanoback/policy.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace nanoback {

namespace {

template <typename T>
[[nodiscard]] std::span<const T> checked_1d_span(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& array,
    const char* name
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string{name} + " must be a 1D contiguous array");
    }
    return {
        static_cast<const T*>(array.data()),
        static_cast<std::size_t>(array.shape(0))
    };
}

template <typename T>
struct MatrixView {
    std::span<const T> values;
    std::size_t rows{0};
    std::size_t cols{0};
};

template <typename T>
[[nodiscard]] MatrixView<T> checked_2d_matrix(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& array,
    const char* name
) {
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string{name} + " must be a 2D contiguous array");
    }
    return MatrixView<T>{
        .values = {
            static_cast<const T*>(array.data()),
            static_cast<std::size_t>(array.shape(0) * array.shape(1))
        },
        .rows = static_cast<std::size_t>(array.shape(0)),
        .cols = static_cast<std::size_t>(array.shape(1)),
    };
}

template <typename T>
[[nodiscard]] py::array_t<T> to_numpy_2d(
    const std::vector<T>& values,
    const std::size_t rows,
    const std::size_t cols
) {
    py::array_t<T> out({rows, cols});
    std::copy(values.begin(), values.end(), static_cast<T*>(out.mutable_data()));
    return out;
}

}  // namespace

PYBIND11_MODULE(_nanoback, module) {
    module.doc() = "C++ core for the nanoback event-driven backtester";

    py::enum_<OrderType>(module, "OrderType")
        .value("MARKET", OrderType::market)
        .value("LIMIT", OrderType::limit);

    py::enum_<SlippageModel>(module, "SlippageModel")
        .value("NONE", SlippageModel::none)
        .value("FIXED_BPS", SlippageModel::fixed_bps)
        .value("VOLUME_SHARE", SlippageModel::volume_share);

    py::enum_<AuditEventType>(module, "AuditEventType")
        .value("ORDER_SUBMITTED", AuditEventType::order_submitted)
        .value("ORDER_REJECTED_LIMIT", AuditEventType::order_rejected_limit)
        .value("ORDER_REJECTED_LEVERAGE", AuditEventType::order_rejected_leverage)
        .value("ORDER_REJECTED_CASH", AuditEventType::order_rejected_cash)
        .value("ORDER_CANCELLED_SESSION", AuditEventType::order_cancelled_session)
        .value("FILL_APPLIED", AuditEventType::fill_applied)
        .value("RISK_KILL_SWITCH", AuditEventType::risk_kill_switch)
        .value("ORDER_WAITING_QUEUE", AuditEventType::order_waiting_queue);

    py::class_<BacktestConfig>(module, "BacktestConfig")
        .def(
            py::init([](double starting_cash,
                        double commission_bps,
                        double slippage_bps,
                        double volume_share_impact,
                        double max_participation_rate,
                        double default_volume,
                        std::int64_t lot_size,
                        std::int64_t max_position,
                        std::int64_t latency_steps,
                        std::int64_t child_order_size,
                        double annual_borrow_bps,
                        double annual_cash_yield_bps,
                        double max_gross_leverage,
                        double max_drawdown_pct,
                        double queue_ahead_fraction,
                        double venue_volume_share_cap,
                        SlippageModel slippage_model,
                        bool allow_short,
                        bool mark_to_market,
                        bool cancel_orders_outside_session,
                        bool use_bid_ask_execution) {
                return BacktestConfig{
                    .starting_cash = starting_cash,
                    .commission_bps = commission_bps,
                    .slippage_bps = slippage_bps,
                    .volume_share_impact = volume_share_impact,
                    .max_participation_rate = max_participation_rate,
                    .default_volume = default_volume,
                    .lot_size = lot_size,
                    .max_position = max_position,
                    .latency_steps = latency_steps,
                    .child_order_size = child_order_size,
                    .annual_borrow_bps = annual_borrow_bps,
                    .annual_cash_yield_bps = annual_cash_yield_bps,
                    .max_gross_leverage = max_gross_leverage,
                    .max_drawdown_pct = max_drawdown_pct,
                    .queue_ahead_fraction = queue_ahead_fraction,
                    .venue_volume_share_cap = venue_volume_share_cap,
                    .slippage_model = slippage_model,
                    .allow_short = allow_short,
                    .mark_to_market = mark_to_market,
                    .cancel_orders_outside_session = cancel_orders_outside_session,
                    .use_bid_ask_execution = use_bid_ask_execution,
                };
            }),
            py::arg("starting_cash") = 1'000'000.0,
            py::arg("commission_bps") = 0.0,
            py::arg("slippage_bps") = 0.0,
            py::arg("volume_share_impact") = 0.05,
            py::arg("max_participation_rate") = 0.25,
            py::arg("default_volume") = 1'000'000.0,
            py::arg("lot_size") = 1,
            py::arg("max_position") = 1,
            py::arg("latency_steps") = 0,
            py::arg("child_order_size") = 0,
            py::arg("annual_borrow_bps") = 0.0,
            py::arg("annual_cash_yield_bps") = 0.0,
            py::arg("max_gross_leverage") = 10.0,
            py::arg("max_drawdown_pct") = 1.0,
            py::arg("queue_ahead_fraction") = 0.0,
            py::arg("venue_volume_share_cap") = 1.0,
            py::arg("slippage_model") = SlippageModel::volume_share,
            py::arg("allow_short") = true,
            py::arg("mark_to_market") = true,
            py::arg("cancel_orders_outside_session") = true,
            py::arg("use_bid_ask_execution") = false
        )
        .def_readwrite("starting_cash", &BacktestConfig::starting_cash)
        .def_readwrite("commission_bps", &BacktestConfig::commission_bps)
        .def_readwrite("slippage_bps", &BacktestConfig::slippage_bps)
        .def_readwrite("volume_share_impact", &BacktestConfig::volume_share_impact)
        .def_readwrite("max_participation_rate", &BacktestConfig::max_participation_rate)
        .def_readwrite("default_volume", &BacktestConfig::default_volume)
        .def_readwrite("lot_size", &BacktestConfig::lot_size)
        .def_readwrite("max_position", &BacktestConfig::max_position)
        .def_readwrite("latency_steps", &BacktestConfig::latency_steps)
        .def_readwrite("child_order_size", &BacktestConfig::child_order_size)
        .def_readwrite("annual_borrow_bps", &BacktestConfig::annual_borrow_bps)
        .def_readwrite("annual_cash_yield_bps", &BacktestConfig::annual_cash_yield_bps)
        .def_readwrite("max_gross_leverage", &BacktestConfig::max_gross_leverage)
        .def_readwrite("max_drawdown_pct", &BacktestConfig::max_drawdown_pct)
        .def_readwrite("queue_ahead_fraction", &BacktestConfig::queue_ahead_fraction)
        .def_readwrite("venue_volume_share_cap", &BacktestConfig::venue_volume_share_cap)
        .def_readwrite("slippage_model", &BacktestConfig::slippage_model)
        .def_readwrite("allow_short", &BacktestConfig::allow_short)
        .def_readwrite("mark_to_market", &BacktestConfig::mark_to_market)
        .def_readwrite("cancel_orders_outside_session", &BacktestConfig::cancel_orders_outside_session)
        .def_readwrite("use_bid_ask_execution", &BacktestConfig::use_bid_ask_execution);

    py::class_<Fill>(module, "Fill")
        .def_readonly("timestamp", &Fill::timestamp)
        .def_readonly("order_id", &Fill::order_id)
        .def_readonly("parent_order_id", &Fill::parent_order_id)
        .def_readonly("asset", &Fill::asset)
        .def_readonly("price", &Fill::price)
        .def_readonly("quantity", &Fill::quantity)
        .def_readonly("remaining_quantity", &Fill::remaining_quantity)
        .def_readonly("fee", &Fill::fee)
        .def_readonly("order_type", &Fill::order_type);

    py::class_<AuditEvent>(module, "AuditEvent")
        .def_readonly("timestamp", &AuditEvent::timestamp)
        .def_readonly("order_id", &AuditEvent::order_id)
        .def_readonly("parent_order_id", &AuditEvent::parent_order_id)
        .def_readonly("asset", &AuditEvent::asset)
        .def_readonly("type", &AuditEvent::type)
        .def_readonly("value", &AuditEvent::value);

    py::class_<LedgerEntry>(module, "LedgerEntry")
        .def_readonly("sequence", &LedgerEntry::sequence)
        .def_readonly("timestamp", &LedgerEntry::timestamp)
        .def_readonly("order_id", &LedgerEntry::order_id)
        .def_readonly("parent_order_id", &LedgerEntry::parent_order_id)
        .def_readonly("asset", &LedgerEntry::asset)
        .def_readonly("type", &LedgerEntry::type)
        .def_readonly("quantity", &LedgerEntry::quantity)
        .def_readonly("remaining_quantity", &LedgerEntry::remaining_quantity)
        .def_readonly("price", &LedgerEntry::price)
        .def_readonly("cash_after", &LedgerEntry::cash_after)
        .def_readonly("equity_after", &LedgerEntry::equity_after)
        .def_readonly("value", &LedgerEntry::value);

    py::class_<BacktestResult>(module, "BacktestResult")
        .def_readonly("ending_cash", &BacktestResult::ending_cash)
        .def_readonly("ending_equity", &BacktestResult::ending_equity)
        .def_readonly("pnl", &BacktestResult::pnl)
        .def_readonly("turnover", &BacktestResult::turnover)
        .def_readonly("total_fees", &BacktestResult::total_fees)
        .def_readonly("total_borrow_cost", &BacktestResult::total_borrow_cost)
        .def_readonly("total_cash_yield", &BacktestResult::total_cash_yield)
        .def_readonly("peak_equity", &BacktestResult::peak_equity)
        .def_readonly("max_drawdown", &BacktestResult::max_drawdown)
        .def_readonly("submitted_orders", &BacktestResult::submitted_orders)
        .def_readonly("filled_orders", &BacktestResult::filled_orders)
        .def_readonly("rejected_orders", &BacktestResult::rejected_orders)
        .def_readonly("halted_by_risk", &BacktestResult::halted_by_risk)
        .def_readonly("equity_curve", &BacktestResult::equity_curve)
        .def_readonly("cash_curve", &BacktestResult::cash_curve)
        .def_property_readonly(
            "positions",
            [](const BacktestResult& self) {
                return py::array_t<std::int64_t>(
                    self.positions.size(),
                    self.positions.data(),
                    py::cast(&self)
                );
            }
        )
        .def_readonly("fills", &BacktestResult::fills)
        .def_readonly("audit_events", &BacktestResult::audit_events)
        .def_readonly("ledger", &BacktestResult::ledger);

    py::class_<Backtester>(module, "Backtester")
        .def(py::init<>())
        .def(
            "run_matrix",
            [](const Backtester& self,
               const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& timestamps,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& high,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& low,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& volume,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& bid,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& ask,
               const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& target_positions,
               const py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>& order_types,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& limit_prices,
               const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& tradable_mask,
               const BacktestConfig& config) {
                const auto ts_view = checked_1d_span(timestamps, "timestamps");
                const auto close_view = checked_2d_matrix(close, "close");
                const auto high_view = checked_2d_matrix(high, "high");
                const auto low_view = checked_2d_matrix(low, "low");
                const auto volume_view = checked_2d_matrix(volume, "volume");
                const auto bid_view = checked_2d_matrix(bid, "bid");
                const auto ask_view = checked_2d_matrix(ask, "ask");
                const auto target_view = checked_2d_matrix(target_positions, "target_positions");
                const auto type_view = checked_2d_matrix(order_types, "order_types");
                const auto limit_view = checked_2d_matrix(limit_prices, "limit_prices");
                const auto tradable_view = checked_1d_span(tradable_mask, "tradable_mask");

                if (close_view.rows != target_view.rows || close_view.cols != target_view.cols) {
                    throw std::invalid_argument("close and target_positions must have the same shape");
                }

                return self.run(
                    ts_view,
                    close_view.values,
                    high_view.values,
                    low_view.values,
                    volume_view.values,
                    bid_view.values,
                    ask_view.values,
                    target_view.values,
                    type_view.values,
                    limit_view.values,
                    tradable_view,
                    close_view.rows,
                    close_view.cols,
                    config
                );
            },
            py::arg("timestamps"),
            py::arg("close"),
            py::arg("high"),
            py::arg("low"),
            py::arg("volume"),
            py::arg("bid"),
            py::arg("ask"),
            py::arg("target_positions"),
            py::arg("order_types"),
            py::arg("limit_prices"),
            py::arg("tradable_mask"),
            py::arg("config") = BacktestConfig{}
        );

    py::class_<PolicyEngine>(module, "PolicyEngine")
        .def(py::init<>())
        .def(
            "rolling_volatility",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t window) {
                const auto close_view = checked_2d_matrix(close, "close");
                return to_numpy_2d(
                    self.rolling_volatility(close_view.values, close_view.rows, close_view.cols, window),
                    close_view.rows,
                    close_view.cols
                );
            },
            py::arg("close"),
            py::arg("window")
        )
        .def(
            "cross_sectional_rank",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
               bool descending) {
                const auto view = checked_2d_matrix(values, "values");
                return to_numpy_2d(
                    self.cross_sectional_rank(view.values, view.rows, view.cols, descending),
                    view.rows,
                    view.cols
                );
            },
            py::arg("values"),
            py::arg("descending") = true
        )
        .def(
            "minimum_variance_weights",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t window,
               double ridge,
               double leverage) {
                const auto close_view = checked_2d_matrix(close, "close");
                return to_numpy_2d(
                    self.minimum_variance_weights(
                        close_view.values,
                        close_view.rows,
                        close_view.cols,
                        window,
                        ridge,
                        leverage
                    ),
                    close_view.rows,
                    close_view.cols
                );
            },
            py::arg("close"),
            py::arg("window"),
            py::arg("ridge") = 1e-6,
            py::arg("leverage") = 1.0
        )
        .def(
            "momentum_targets",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t lookback,
               std::int64_t max_position) {
                const auto close_view = checked_2d_matrix(close, "close");
                auto targets = self.momentum_targets(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    lookback,
                    max_position
                );
                return to_numpy_2d(targets, close_view.rows, close_view.cols);
            },
            py::arg("close"),
            py::arg("lookback"),
            py::arg("max_position")
        )
        .def(
            "mean_reversion_targets",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t lookback,
               std::int64_t max_position) {
                const auto close_view = checked_2d_matrix(close, "close");
                auto targets = self.mean_reversion_targets(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    lookback,
                    max_position
                );
                return to_numpy_2d(targets, close_view.rows, close_view.cols);
            },
            py::arg("close"),
            py::arg("lookback"),
            py::arg("max_position")
        )
        .def(
            "moving_average_crossover_targets",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t fast_window,
               std::size_t slow_window,
               std::int64_t max_position) {
                const auto close_view = checked_2d_matrix(close, "close");
                auto targets = self.moving_average_crossover_targets(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    fast_window,
                    slow_window,
                    max_position
                );
                return to_numpy_2d(targets, close_view.rows, close_view.cols);
            },
            py::arg("close"),
            py::arg("fast_window"),
            py::arg("slow_window"),
            py::arg("max_position")
        )
        .def(
            "volatility_filtered_momentum_targets",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t lookback,
               std::size_t vol_window,
               double volatility_ceiling,
               std::int64_t max_position) {
                const auto close_view = checked_2d_matrix(close, "close");
                return to_numpy_2d(
                    self.volatility_filtered_momentum_targets(
                        close_view.values,
                        close_view.rows,
                        close_view.cols,
                        lookback,
                        vol_window,
                        volatility_ceiling,
                        max_position
                    ),
                    close_view.rows,
                    close_view.cols
                );
            },
            py::arg("close"),
            py::arg("lookback"),
            py::arg("vol_window"),
            py::arg("volatility_ceiling"),
            py::arg("max_position")
        )
        .def(
            "cross_sectional_momentum_targets",
            [](const PolicyEngine& self,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
               std::size_t lookback,
               std::size_t winners,
               std::size_t losers,
               std::int64_t max_position) {
                const auto close_view = checked_2d_matrix(close, "close");
                return to_numpy_2d(
                    self.cross_sectional_momentum_targets(
                        close_view.values,
                        close_view.rows,
                        close_view.cols,
                        lookback,
                        winners,
                        losers,
                        max_position
                    ),
                    close_view.rows,
                    close_view.cols
                );
            },
            py::arg("close"),
            py::arg("lookback"),
            py::arg("winners"),
            py::arg("losers"),
            py::arg("max_position")
        );

    module.def(
        "run_backtest_matrix",
        [](const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& timestamps,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& high,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& low,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& volume,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& bid,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& ask,
           const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& target_positions,
           const py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>& order_types,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& limit_prices,
           const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& tradable_mask,
           const BacktestConfig& config) {
            Backtester backtester;
            const auto ts_view = checked_1d_span(timestamps, "timestamps");
            const auto close_view = checked_2d_matrix(close, "close");
            const auto high_view = checked_2d_matrix(high, "high");
            const auto low_view = checked_2d_matrix(low, "low");
            const auto volume_view = checked_2d_matrix(volume, "volume");
            const auto bid_view = checked_2d_matrix(bid, "bid");
            const auto ask_view = checked_2d_matrix(ask, "ask");
            const auto target_view = checked_2d_matrix(target_positions, "target_positions");
            const auto type_view = checked_2d_matrix(order_types, "order_types");
            const auto limit_view = checked_2d_matrix(limit_prices, "limit_prices");
            const auto tradable_view = checked_1d_span(tradable_mask, "tradable_mask");

            return backtester.run(
                ts_view,
                close_view.values,
                high_view.values,
                low_view.values,
                volume_view.values,
                bid_view.values,
                ask_view.values,
                target_view.values,
                type_view.values,
                limit_view.values,
                tradable_view,
                close_view.rows,
                close_view.cols,
                config
            );
        },
        py::arg("timestamps"),
        py::arg("close"),
        py::arg("high"),
        py::arg("low"),
        py::arg("volume"),
        py::arg("bid"),
        py::arg("ask"),
        py::arg("target_positions"),
        py::arg("order_types"),
        py::arg("limit_prices"),
        py::arg("tradable_mask"),
        py::arg("config") = BacktestConfig{}
    );

    module.def(
        "rolling_volatility",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t window) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            return to_numpy_2d(
                engine.rolling_volatility(close_view.values, close_view.rows, close_view.cols, window),
                close_view.rows,
                close_view.cols
            );
        },
        py::arg("close"),
        py::arg("window")
    );

    module.def(
        "cross_sectional_rank",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
           bool descending) {
            PolicyEngine engine;
            const auto view = checked_2d_matrix(values, "values");
            return to_numpy_2d(
                engine.cross_sectional_rank(view.values, view.rows, view.cols, descending),
                view.rows,
                view.cols
            );
        },
        py::arg("values"),
        py::arg("descending") = true
    );

    module.def(
        "minimum_variance_weights",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t window,
           double ridge,
           double leverage) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            return to_numpy_2d(
                engine.minimum_variance_weights(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    window,
                    ridge,
                    leverage
                ),
                close_view.rows,
                close_view.cols
            );
        },
        py::arg("close"),
        py::arg("window"),
        py::arg("ridge") = 1e-6,
        py::arg("leverage") = 1.0
    );

    module.def(
        "momentum_targets",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t lookback,
           std::int64_t max_position) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            auto targets = engine.momentum_targets(
                close_view.values,
                close_view.rows,
                close_view.cols,
                lookback,
                max_position
            );
            return to_numpy_2d(targets, close_view.rows, close_view.cols);
        },
        py::arg("close"),
        py::arg("lookback"),
        py::arg("max_position")
    );

    module.def(
        "mean_reversion_targets",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t lookback,
           std::int64_t max_position) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            auto targets = engine.mean_reversion_targets(
                close_view.values,
                close_view.rows,
                close_view.cols,
                lookback,
                max_position
            );
            return to_numpy_2d(targets, close_view.rows, close_view.cols);
        },
        py::arg("close"),
        py::arg("lookback"),
        py::arg("max_position")
    );

    module.def(
        "moving_average_crossover_targets",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t fast_window,
           std::size_t slow_window,
           std::int64_t max_position) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            auto targets = engine.moving_average_crossover_targets(
                close_view.values,
                close_view.rows,
                close_view.cols,
                fast_window,
                slow_window,
                max_position
            );
            return to_numpy_2d(targets, close_view.rows, close_view.cols);
        },
        py::arg("close"),
        py::arg("fast_window"),
        py::arg("slow_window"),
        py::arg("max_position")
    );

    module.def(
        "volatility_filtered_momentum_targets",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t lookback,
           std::size_t vol_window,
           double volatility_ceiling,
           std::int64_t max_position) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            return to_numpy_2d(
                engine.volatility_filtered_momentum_targets(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    lookback,
                    vol_window,
                    volatility_ceiling,
                    max_position
                ),
                close_view.rows,
                close_view.cols
            );
        },
        py::arg("close"),
        py::arg("lookback"),
        py::arg("vol_window"),
        py::arg("volatility_ceiling"),
        py::arg("max_position")
    );

    module.def(
        "cross_sectional_momentum_targets",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& close,
           std::size_t lookback,
           std::size_t winners,
           std::size_t losers,
           std::int64_t max_position) {
            PolicyEngine engine;
            const auto close_view = checked_2d_matrix(close, "close");
            return to_numpy_2d(
                engine.cross_sectional_momentum_targets(
                    close_view.values,
                    close_view.rows,
                    close_view.cols,
                    lookback,
                    winners,
                    losers,
                    max_position
                ),
                close_view.rows,
                close_view.cols
            );
        },
        py::arg("close"),
        py::arg("lookback"),
        py::arg("winners"),
        py::arg("losers"),
        py::arg("max_position")
    );
}

}  // namespace nanoback
