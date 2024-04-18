from typing import Optional
from typing_extensions import Self
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory
from gaitmap._event_detection_common._event_detection_mixin import _EventDetectionMixin

# from gaitmap.base import BaseStrideSegmentation
from gaitmap.utils.datatype_helper import SensorData


class HoriStrideSegmentation(_EventDetectionMixin):
    """Segment the sensor into consecutive strides."""

    strides_: pd.DataFrame
    memory: Optional[Memory]

    def __init__(self, memory: Optional[Memory] = None) -> None:
        super().__init__(memory=memory)

    def detect(
        self,
        data: SensorData,
        *,
        sampling_rate_hz: float,
        gait_events: pd.DataFrame,
        visualize: bool = False
    ) -> Self:
        """Determine the start and end of each stride."""

        # Extract the mediolateral angular velocity
        gyr_ml = data["gyr_y"].to_numpy()

        # Loop over the strides
        idx_start = np.full((len(gait_events),), fill_value=np.nan)
        idx_end = np.full((len(gait_events),), fill_value=np.nan)
        for i in range(len(gait_events) - 1):
            idx_ic = gait_events["ic"][i].astype(
                int
            )  # the initial contact of the current stride
            idx_fc = gait_events["fc"][i + 1].astype(
                int
            )  # the final contact of the next stride

            # Fit a quadratic curve to the signal segment
            coeffs = np.polyfit(np.arange(idx_ic, idx_fc), gyr_ml[idx_ic:idx_fc], deg=2)
            p = np.poly1d(coeffs)  # instantiate poly1d object
            curve = p(np.arange(idx_ic, idx_fc))  # fit a local curve
            idx_min = np.argmin(curve)  # find the minimum

            # Assign index of local minimum to end of current
            # and start of next stride
            idx_end[i] = idx_ic + idx_min
            idx_start[i + 1] = idx_ic + idx_min

        # Add to a copy of the events dataframe
        self.strides_ = gait_events.copy()
        self.strides_["start"] = idx_start
        self.strides_["min_vel"] = self.strides_["start"]
        self.strides_["end"] = idx_end
        self.strides_["s_id"] = range(len(self.strides_))
        self.strides_ = self.strides_[
            ["s_id", "start", "min_vel", "fc", "ms", "ic", "end"]
        ]
        return self
