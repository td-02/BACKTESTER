#pragma once

#include <cstdint>
#include <vector>

namespace nanoback {

struct Venue {
    std::int64_t venue_id{0};
    double maker_fee_bps{0.0};
    double taker_fee_bps{0.0};
    std::int64_t one_way_latency_us{0};
    double volume_share{1.0};
    std::vector<double> fill_probability_curve{1.0};
};

}  // namespace nanoback

