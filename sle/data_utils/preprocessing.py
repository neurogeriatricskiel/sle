import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
from gaitmap.utils.datatype_helper import SensorData
from gaitmap.utils import rotations


logger = logging.getLogger(__name__)


def _find_adaptive_threshold(
    signal: np.ndarray,
    weighting_factor: float = 0.85,
    num_iter: int = 200,
    thr_min: float = 0.2,
) -> float:
    """Find threshold in an iterative manner. Based on Laidig et al., (2012).

    Parameters
    ----------
    signal : np.ndarray
        The signal.
    weighting_factor : float
        A weighting parameter.
    num_iter : int, optional
        The number of iterations, by default 200
    thr_min: float
        Lower bound for the threshold value.

    Returns
    -------
    float
        The final threshold.
    """

    # Calculate initial threshold, Eqn (5)
    thr = 0.5 * (max(signal) + min(signal))

    for i in range(num_iter):
        idx_pos = np.argwhere(signal > thr)[:, 0]  # Eqn (6)
        idx_neg = np.argwhere(signal <= thr)[:, 0]  # Eqn (7)
        thr = (weighting_factor / len(idx_neg)) * np.sum(signal[idx_neg]) + (
            (1 - weighting_factor) / len(idx_pos)
        ) * np.sum(signal[idx_pos])
    return np.max((thr, thr_min))


def _determine_rest_signal(
    signal: np.ndarray,
    sampling_frequency_Hz: float,
    thr: float,
    hysteresis_factor: float = 0.23,
    T_0_min_ms: float = 120.0,
    T_1_min_ms: float = 180.0,
) -> np.ndarray:
    """Determine the rest signal, which is low (0) when at rest and high (1) when moving.

    Parameters
    ----------
    signal : np.ndarray
        The signal.
    sampling_frequency_Hz : float
        The sampling frequency (Hz) at which the signal was recorded.
    thr : float
        The threshold value to determine stationary or moving periods.
    hysteresis_factor : float, optional
        The hysteresis factor, by default 0.23
    T_0_min_ms : float, optional
        The minimum duration of a zero (stationary) phase, by default 120.0
    T_1_min_ms : float, optional
        The minimum duration of a one (moving) phase, by default 180.0

    Returns
    -------
    np.ndarray
        The resulting signal that signals stationarity or moving.
    """
    # Convert minimum durations to an integer number of samples
    T_0_min = int(T_0_min_ms / 1000.0 * sampling_frequency_Hz)
    T_1_min = int(T_1_min_ms / 1000.0 * sampling_frequency_Hz)

    # Determine rest signal - forward iteration
    r_a_init = np.zeros_like(signal, dtype=int)
    for i in range(1, len(signal)):
        if signal[i] > (1 + hysteresis_factor) * thr:
            r_a_init[i] = 1
        elif signal[i] < (1 - hysteresis_factor) * thr:
            r_a_init[i] = 0
        else:
            r_a_init[i] = r_a_init[i - 1]

    # Backward iteration
    r_a = r_a_init.copy()
    for i in range(len(r_a) - 2, -1, -1):
        if r_a_init[i] == 1:
            r_a[i] = 1
        elif signal[i] < (1 - hysteresis_factor) * thr:
            r_a[i] = 0
        else:
            r_a[i] = r_a_init[i + 1]

    # In the resulting signal, zero-phases shorter than T_0_min are set to one
    idx_pos_neg = np.argwhere(np.diff(r_a) < 0)[:, 0]
    idx_neg_pos = np.argwhere(np.diff(r_a) > 0)[:, 0]
    for i in range(len(idx_neg_pos)):
        f = np.argwhere(idx_pos_neg < idx_neg_pos[i])[:, 0]
        if len(f) > 0:
            if idx_neg_pos[i] + 1 - idx_pos_neg[f[-1]] < T_0_min:
                r_a[idx_pos_neg[f[-1]] : idx_neg_pos[i] + 1] = 1

    # and afterward, one-phases shorter than T1,min are set to zero
    idx_pos_neg = np.argwhere(np.diff(r_a) < 0)[:, 0]
    idx_neg_pos = np.argwhere(np.diff(r_a) > 0)[:, 0]
    for i in range(len(idx_pos_neg)):
        f = np.argwhere(idx_neg_pos < idx_pos_neg[i])[:, 0]
        if len(f) > 0:
            if idx_pos_neg[i] + 1 - idx_neg_pos[f[-1]] < T_1_min:
                r_a[idx_neg_pos[f[-1]] : idx_pos_neg[i] + 1] = 0
    return r_a


def _is_stationary(
    acc: np.ndarray,
    gyr: np.ndarray,
    sampling_frequency_Hz: float,
    weighting_factor: float = 0.9,
    hysteresis_factor: float = 0.15,
    T_0_min_ms: float = 120.0,
    T_1_min_ms: float = 180.0,
) -> np.ndarray:
    # Convert minimum durations to an integer number of samples
    T_0_min = int(T_0_min_ms / 1000.0 * sampling_frequency_Hz)
    T_1_min = int(T_1_min_ms / 1000.0 * sampling_frequency_Hz)

    # Calculate the norm of the signals
    acc_norm = np.abs(np.linalg.norm(acc, axis=-1) - 9.81)
    gyr_norm = np.abs(np.linalg.norm(gyr, axis=-1))

    # Find adaptive thresholds
    thr_acc = _find_adaptive_threshold(acc_norm, weighting_factor=weighting_factor)
    thr_gyr = _find_adaptive_threshold(
        gyr_norm, weighting_factor=weighting_factor, thr_min=0.1
    )

    # Determine the rest signals
    acc_rest = _determine_rest_signal(
        acc_norm,
        sampling_frequency_Hz=sampling_frequency_Hz,
        thr=thr_acc,
        hysteresis_factor=hysteresis_factor,
    )
    gyr_rest = _determine_rest_signal(
        gyr_norm,
        sampling_frequency_Hz=sampling_frequency_Hz,
        thr=thr_gyr,
        hysteresis_factor=hysteresis_factor,
    )

    # Both rest signals, ra(tk) and rω(tk), are combined into r(tk),
    # which is set to one if at least one of the two signals is one.
    rest = acc_rest | gyr_rest

    # Afterward, zero-phases shorter than T_0_min are set to one,
    idx_pos_neg = np.argwhere(np.diff(rest) < 0)[:, 0]
    idx_neg_pos = np.argwhere(np.diff(rest) > 0)[:, 0]
    for i in range(len(idx_neg_pos)):
        f = np.argwhere(idx_pos_neg < idx_neg_pos[i])[:, 0]
        if len(f) > 0:
            if idx_neg_pos[i] + 1 - idx_pos_neg[f[-1]] < T_0_min:
                rest[idx_pos_neg[f[-1]] : idx_neg_pos[i] + 1] = 1

    # and then one-phases shorter than 2 T_1_min are set to zero.
    idx_pos_neg = np.argwhere(np.diff(rest) < 0)[:, 0]
    idx_neg_pos = np.argwhere(np.diff(rest) > 0)[:, 0]
    for i in range(len(idx_pos_neg)):
        f = np.argwhere(idx_neg_pos < idx_pos_neg[i])[:, 0]
        if len(f) > 0:
            if idx_pos_neg[i] + 1 - idx_neg_pos[f[-1]] < (2 * T_1_min):
                rest[idx_neg_pos[f[-1]] : idx_pos_neg[i] + 1] = 0
    return rest


def align_with_gravity(
    dataset: SensorData,
    sampling_frequency_Hz: float,
    min_stationary_time_ms: float = 500.0,
    weighting_factor: float = 0.9,
    hysteresis_factor: float = 0.15,
    T_0_min_ms: float = 120.0,
    T_1_min_ms: float = 180.0,
) -> SensorData:
    # Convert minimum stationary time from ms to samples
    min_stationary_time = int(min_stationary_time_ms / 1000.0 * sampling_frequency_Hz)

    # Get acceleration and gyroscope data
    acc = dataset[[f"acc_{x}" for x in ["x", "y", "z"]]]
    gyr = dataset[[f"gyr_{x}" for x in ["x", "y", "z"]]]

    # Determine the stationary periods
    rest = _is_stationary(
        acc=acc,
        gyr=gyr,
        sampling_frequency_Hz=sampling_frequency_Hz,
        weighting_factor=weighting_factor,
        hysteresis_factor=hysteresis_factor,
        T_0_min_ms=T_0_min_ms,
        T_1_min_ms=T_1_min_ms,
    )

    # If trial starts with stationary period,
    # and lasts for at least `min_stationary_time_ms`
    if rest[0] != 0:
        logging.info("... ... No initial stationary period was detected!")
        return
    f = np.argwhere(rest == 1)[:, 0]  # find first non-stationary sample
    if len(f) > 0:
        idx = f[0]
    else:
        logging.info("... ... The entire trial was stationary!")
        return

    # Get rotation
    rotation = rotations.get_gravity_rotation(acc.iloc[:idx].median(axis=0))
    return rotations.rotate_dataset(dataset, rotation)
