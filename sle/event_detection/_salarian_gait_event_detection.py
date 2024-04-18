from typing import Optional, Tuple
from typing_extensions import Self
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory
from gaitmap._event_detection_common._event_detection_mixin import _EventDetectionMixin
from gaitmap.base import BaseEventDetection
from gaitmap.utils.datatype_helper import SensorData


class SalarianGaitEventDetection(_EventDetectionMixin, BaseEventDetection):
    """Find gait events in the ankle- or shank-worn IMU signals."""

    min_peak_angular_velocity: float
    min_peak_distance_ms: float
    min_fc_angular_velocity: float
    gait_events_: pd.DataFrame
    memory: Optional[Memory]

    def __init__(
        self,
        min_peak_angular_velocity: float = 50.0,
        min_peak_distance_ms: float = 500.0,
        min_fc_angular_velocity: float = 20.0,
        memory: Optional[Memory] = None,
    ) -> None:
        self.min_peak_angular_velocity = min_peak_angular_velocity
        self.min_peak_distance_ms = min_peak_distance_ms
        self.min_fc_angular_velocity = min_fc_angular_velocity
        super().__init__(memory=memory)

    def _detect_midswings(self, gyr_ml: np.ndarray) -> np.ndarray:
        """Detect midswings from the mediolateral angular velocity signal."""
        min_peak_distance = int(
            self.min_peak_distance_ms / 1000 * self.sampling_rate_hz
        )
        ipks, _ = signal.find_peaks(
            -gyr_ml, height=self.min_peak_angular_velocity, distance=min_peak_distance
        )
        return ipks

    def _detect_initial_and_final_contacts(
        self, gyr_ml: np.ndarray, idx_midswings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect initial and final contacts from the mediolateral angular velocity signal."""
        # Detect local maxima around the indices of midswings
        idx_max, _ = signal.find_peaks(gyr_ml, height=0.0)

        # Pre-allocate output arrays
        idx_ics = np.full_like(idx_midswings, fill_value=np.nan, dtype=float)
        idx_fcs = np.full_like(idx_midswings, fill_value=np.nan, dtype=float)

        # Loop over the indices of midswings
        for i in range(len(idx_midswings)):

            # Find the nearest local maximum after the current midswing
            f = np.argwhere(idx_max > idx_midswings[i])[:, 0]
            if len(f) > 0:
                idx_ics[i] = idx_max[f[0]]

            # Find the local maximum prior to the current midswing
            # that has an amplitude greater than a given minimum angular velocity for final contacts detection
            f = np.argwhere(
                (idx_max < idx_midswings[i])
                & (gyr_ml[idx_max] > self.min_fc_angular_velocity)
            )[:, 0]
            if len(f) > 0:
                idx_fcs[i] = idx_max[f[-1]]
        return idx_ics, idx_fcs

    def detect(
        self, data: SensorData, *, sampling_rate_hz: float, visualize: bool = False
    ) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        gyr_ml = data["gyr_y"].to_numpy()
        idx_midswings = self._detect_midswings(gyr_ml=gyr_ml)
        idx_ics, idx_fcs = self._detect_initial_and_final_contacts(
            gyr_ml=gyr_ml, idx_midswings=idx_midswings
        )

        self.gait_events_ = pd.DataFrame(
            {
                "s_id": np.arange(len(idx_midswings)),
                "fc": idx_fcs,
                "ms": idx_midswings,
                "ic": idx_ics,
            }
        )

        if visualize:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(gyr_ml)), gyr_ml, c="tab:blue", lw=2, alpha=0.5)
            ax.plot(
                idx_midswings,
                gyr_ml[idx_midswings],
                ls="none",
                marker="o",
                mfc="none",
                mec="tab:blue",
            )
            ax.plot(
                idx_ics,
                gyr_ml[idx_ics],
                ls="none",
                marker="o",
                mfc="none",
                mec="tab:red",
            )
            ax.plot(
                idx_fcs,
                gyr_ml[idx_fcs],
                ls="none",
                marker="o",
                mfc="none",
                mec="tab:green",
            )
            ax.grid(which="both", axis="both", c="tab:gray", ls=":", alpha=0.2)
            plt.tight_layout()
            plt.show()
        return self
