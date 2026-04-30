#pragma once

#include <cstdint>

namespace nanoback {

enum class CorporateActionType : std::int8_t {
    split = 0,
    dividend = 1,
    spinoff = 2,
    delisting = 3,
};

struct CorporateAction {
    std::int64_t asset{0};
    std::int64_t ex_date_timestamp{0};
    CorporateActionType action_type{CorporateActionType::split};
    double ratio_or_amount{0.0};
};

}  // namespace nanoback

