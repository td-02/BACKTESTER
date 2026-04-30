#pragma once

#include <cstdint>

namespace nanoback {

enum class TickSide : std::int8_t {
    bid = 0,
    ask = 1,
    trade = 2,
};

struct TickEvent {
    std::int64_t timestamp_ns{0};
    std::int64_t asset{0};
    double price{0.0};
    double size{0.0};
    TickSide side{TickSide::trade};
};

}  // namespace nanoback

