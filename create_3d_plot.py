import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from sle.data_utils import data_loader

from sle.data_utils import data_loader
from sle.data_utils.preprocessing import (
    align_with_gravity,
    align_with_forward_direction,
)
from gaitmap.utils.datatype_helper import SensorData
from gaitmap.data_transform import ButterworthFilter
from gaitmap.preprocessing.sensor_alignment import (
    PcaAlignment
)

def preprocess_dataset(dataset: SensorData) -> SensorData:
    
    # Low-pass filter the sensor data
    lowpass_filter = ButterworthFilter(order=4, cutoff_freq_hz=5.0)
    filtered_data = {
        s: lowpass_filter.transform(
            data=dataset[s], sampling_rate_hz=data_loader.SAMPLING_FREQUENCY_HZ
        ).transformed_data_
        for s in dataset.keys()
    }

    # Align the sensor with earth gravity
    gravity_aligned_data = {
        sensor: align_with_gravity(
            filtered_data[sensor],
            sampling_frequency_Hz=data_loader.SAMPLING_FREQUENCY_HZ,
            hysteresis_factor=0.1,
            weighting_factor=0.9,
        )
        for sensor in filtered_data.keys()
    }

    # Align the sensor with walking direction
    walking_aligned_data = {
        sensor: PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y"))
        .align(gravity_aligned_data[sensor])
        .aligned_data_
        for sensor in gravity_aligned_data.keys()
        if gravity_aligned_data[sensor] is not None
    }

    # Align the sensor with forward direction
    forward_aligned_data = {
        sensor: align_with_forward_direction(
            dataset=walking_aligned_data[sensor],
            sampling_frequency_Hz=data_loader.SAMPLING_FREQUENCY_HZ,
        )
        for sensor in walking_aligned_data.keys()
    }
    return forward_aligned_data

from sle.stride_segmentation import HoriStrideSegmentation
from sle.event_detection import SalarianGaitEventDetection
from gaitmap.trajectory_reconstruction import (
    MadgwickAHRS,
    ForwardBackwardIntegration,
    StrideLevelTrajectory,
)

def detect_strides(dataset: SensorData) -> dict[str, pd.DataFrame]:
    # Events detection
    gait_events = {
        sensor: SalarianGaitEventDetection()
        .detect(
            data=dataset[sensor], sampling_rate_hz=data_loader.SAMPLING_FREQUENCY_HZ
        )
        .gait_events_
        for sensor in dataset.keys()
        if dataset[sensor] is not None
    }

    # Stride segmentation
    strides = {
        sensor: HoriStrideSegmentation()
        .detect(
            data=dataset[sensor],
            sampling_rate_hz=data_loader.SAMPLING_FREQUENCY_HZ,
            gait_events=gait_events[sensor],
        )
        .strides_
        for sensor in dataset.keys()
        if dataset[sensor] is not None
    }
    return strides

def estimate_trajectories(
    dataset: SensorData,
    strides: dict[str, pd.DataFrame],
    sampling_frequency_Hz: float = data_loader.SAMPLING_FREQUENCY_HZ,
) -> dict[str, pd.DataFrame]:
    # Reconstruct the trajectory
    ori_method = MadgwickAHRS()
    pos_method = ForwardBackwardIntegration()
    trajectory = StrideLevelTrajectory(ori_method=ori_method, pos_method=pos_method)

    trajectories = {
        sensor: trajectory.estimate(
            data=dataset[sensor],
            stride_event_list=strides[sensor].dropna(axis=0),
            sampling_rate_hz=sampling_frequency_Hz,
        ).position_
        for sensor in dataset.keys()
        if dataset[sensor] is not None
    }
    return trajectories

def main() -> None:
    sub_id = "pp151"
    task = "walkSlow"
    side = "right"
    sub_file_name = pathlib.Path("Z:\\Keep Control\\Data\\lab dataset\\rawdata\\sub-pp151\\motion\\sub-pp151_task-walkSlow_events.tsv")
    marker_dataset = data_loader.load_marker_data(
        sub_file_name.parent
        / sub_file_name.name.replace("_events.tsv", "_tracksys-omc_motion.tsv"),
        tracked_points=["l_heel", "l_ank", "l_toe", "r_heel", "r_ank", "r_toe"],
    )
    imu_dataset = data_loader.load_imu_data(
        sub_file_name.parent
        / sub_file_name.name.replace("_events.tsv", "_tracksys-imu_motion.tsv"),
        tracked_points=["left_ankle", "right_ankle"],
    )

    # Preprocess the IMU dataset
    preprocessed_dataset = preprocess_dataset(dataset=imu_dataset)

    # Detect strides
    strides = detect_strides(dataset=preprocessed_dataset)

    # Reconstruct the trajectories
    trajectories = estimate_trajectories(dataset=preprocessed_dataset, strides=strides)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for s_id, group in trajectories[f"{side}_ankle"].groupby(level="s_id"):
        ax.plot(
            group["pos_x"], group["pos_y"], group["pos_z"]
        )
    ax.set_xlabel("anteroposterior displacement (m)")
    ax.set_ylabel("mediolateral displacement (m)")
    ax.set_zlabel("vertical displacement (m)")
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    main()