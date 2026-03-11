from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SessionCalendar:
    start_minute: int = 9 * 60 + 30
    end_minute: int = 16 * 60
    weekdays: set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})
    holidays: set[int] = field(default_factory=set)

    def tradable_mask(self, timestamps: np.ndarray) -> np.ndarray:
        values = np.asarray(timestamps, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("timestamps must be 1D")

        seconds_in_day = 86_400
        days = np.floor_divide(values, seconds_in_day)
        minute_of_day = np.floor_divide(np.mod(values, seconds_in_day), 60)
        weekday = np.mod(days + 3, 7)  # 1970-01-01 was Thursday.

        allowed_weekdays = np.isin(weekday, np.asarray(sorted(self.weekdays), dtype=np.int64))
        allowed_minutes = (minute_of_day >= self.start_minute) & (minute_of_day <= self.end_minute)
        allowed_holidays = ~np.isin(days, np.asarray(sorted(self.holidays), dtype=np.int64))
        return np.asarray(allowed_weekdays & allowed_minutes & allowed_holidays, dtype=np.uint8)
