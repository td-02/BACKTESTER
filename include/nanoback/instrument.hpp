#pragma once

#include <cstdint>

namespace nanoback {

enum class InstrumentType : std::int8_t {
    equity = 0,
    option_call = 1,
    option_put = 2,
    future = 3,
    fx_forward = 4,
};

struct Instrument {
    InstrumentType type{InstrumentType::equity};
    std::int64_t expiry_timestamp{0};
    double strike{0.0};
    std::int64_t underlying_asset{-1};
    double margin_ratio{0.0};
};

struct FutureRoll {
    std::int64_t from_asset{0};
    std::int64_t to_asset{0};
    std::int64_t roll_timestamp{0};
    double roll_slippage_bps{0.0};
};

}  // namespace nanoback

