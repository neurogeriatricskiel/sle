import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sle.data_utils import data_loader
from sle.data_utils.preprocessing import (
    align_with_gravity,
    align_with_forward_direction,
)
from sle.stride_segmentation import HoriStrideSegmentation
from sle.event_detection import SalarianGaitEventDetection
from gaitmap.utils.datatype_helper import SensorData
from gaitmap.data_transform import ButterworthFilter
from gaitmap.preprocessing.sensor_alignment import (
    PcaAlignment,
    ForwardDirectionSignAlignment,
)
from gaitmap.trajectory_reconstruction import (
    MadgwickAHRS,
    ForwardBackwardIntegration,
    StrideLevelTrajectory,
)


SKIP_FILES = [
    "sub-pp102_task-walkFast_run-off_events.tsv",
    "sub-pp102_task-walkPreferred_run-off_events.tsv",
    "sub-pp102_task-walkSlow_run-off_events.tsv",
]


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    filemode="w",
    format="%(asctime)s %(message)s",
    datefmt="%d %b %Y %H:%M:%S",
    level=logging.DEBUG,
)


def preprocess_dataset(dataset: SensorData) -> SensorData:
    logging.info(f"... ... Preprocessing the IMU data.")

    # Low-pass filter the sensor data
    lowpass_filter = ButterworthFilter(order=4, cutoff_freq_hz=15.0)
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
    logging.info(f"Project's root path: {data_loader.ROOT_PATH}`.")

    demographics_df = data_loader.load_demographics("parkinson_participants.csv")
    results_dict = {
        k: []
        for k in [
            "sub_id",
            "task",
            "run",
            "side",
            "stride_id",
            "stride_length_ref_m",
            "stride_length_pred_m",
        ]
    }
    for sub_id in demographics_df["sub_id"].unique():
        logging.info(f"{'='*60:s}")
        logging.info(f"Processing data from `{sub_id:s}`.")
        event_files = [
            f
            for f in data_loader.ROOT_PATH.joinpath(f"sub-{sub_id}", "motion").iterdir()
            if f.name.endswith("_events.tsv") and "_task-walk" in f.name
        ]

        for event_file in event_files:
            if event_file.name in SKIP_FILES:
                continue  # to next event file
            logging.info(f"... Getting IMU data from `{event_file.name:s}`.")
            if "_run-" in event_file.name:
                run_name = event_file.name[
                    event_file.name.find("_run-") + len("_run-") : -11
                ]
                task_name = event_file.name[
                    event_file.name.find("_task-")
                    + len("_task-") : event_file.name.find("_run-")
                ]
            else:
                run_name = "on"
                task_name = event_file.name[
                    event_file.name.find("_task-") + len("_task-") : -11
                ]

            imu_dataset = data_loader.load_imu_data(
                event_file.parent
                / event_file.name.replace("_events.tsv", "_tracksys-imu_motion.tsv"),
                tracked_points=["left_ankle", "right_ankle"],
            )

            # Preprocessed IMU data
            preprocessed_dataset = preprocess_dataset(dataset=imu_dataset)
            if not preprocessed_dataset:
                logging.info("... ... Continue with next file.")
                continue

            # Detect and segment strides
            strides = detect_strides(dataset=preprocessed_dataset)

            # Estimate the trajectories
            trajectories = estimate_trajectories(
                dataset=preprocessed_dataset, strides=strides
            )

            # Get marker data
            marker_dataset = data_loader.load_marker_data(
                event_file.parent
                / event_file.name.replace("_events.tsv", "_tracksys-omc_motion.tsv"),
                tracked_points=["l_heel", "l_ank", "l_toe", "r_heel", "r_ank", "r_toe"],
            )

            # Print to user screen
            for tracked_point in trajectories.keys():
                # Extract the side, i.e., left or right
                side = tracked_point.replace("_ankle", "")
                if trajectories[tracked_point] is not None:
                    for s_id, group in trajectories[tracked_point].groupby(
                        level="s_id"
                    ):
                        idx_start = (
                            strides[tracked_point].loc[s_id, "start"].astype(int)
                        )
                        idx_end = strides[tracked_point].loc[s_id, "end"].astype(int)

                        # Reference system
                        dx_ref = (
                            marker_dataset[f"{side[:1]}_ank"].loc[idx_end]["pos_x"]
                            - marker_dataset[f"{side[:1]}_ank"].loc[idx_start]["pos_x"]
                        )
                        dy_ref = (
                            marker_dataset[f"{side[:1]}_ank"].loc[idx_end]["pos_y"]
                            - marker_dataset[f"{side[:1]}_ank"].loc[idx_start]["pos_y"]
                        )
                        sl_ref = np.sqrt(dx_ref**2 + dy_ref**2)

                        # Proposed algorithm
                        dx = group["pos_x"].iloc[-1] - group["pos_x"].iloc[0]
                        dy = group["pos_y"].iloc[-1] - group["pos_y"].iloc[0]
                        sl = np.sqrt(dx**2 + dy**2)

                        print(
                            f"{sub_id:<8s}{task_name:<16s}{run_name:<8s}{side:<8s}{s_id:>4d}{sl_ref:>8.2f}{sl:>8.2f}"
                        )
                        for k, val in zip(
                            results_dict.keys(),
                            [sub_id, task_name, run_name, side, s_id, sl_ref, sl],
                        ):
                            results_dict[k].append(val)

    results_df = pd.DataFrame(results_dict)
    results_df = pd.merge(
        results_df,
        demographics_df[
            [
                "sub_id",
                "med_state",
                "gender",
                "age_years",
                "height_cm",
                "weight_kg",
                "bmi",
                "foot_length_cm",
            ]
        ],
        left_on=["sub_id", "run"],
        right_on=["sub_id", "med_state"]
    )
    results_df = results_df[
        [
            "sub_id",
            "gender",
            "age_years",
            "height_cm",
            "weight_kg",
            "bmi",
            "foot_length_cm",
        ]
        + [c for c in results_df.columns if c not in demographics_df.columns]
    ]
    results_df.to_csv("results_ank_markers_15Hz.csv", sep=",", header=True, index=False)
    return


if __name__ == "__main__":
    main()
