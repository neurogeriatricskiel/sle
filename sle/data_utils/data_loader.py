import numpy as np
import pandas as pd
import pathlib
from typing import Optional
from gaitmap.utils.datatype_helper import SensorData


ROOT_PATH = pathlib.Path("Z:\\Keep Control\\Data\\lab dataset\\rawdata")

MAPPING_CHANNEL_TYPES_COLS = {"acc": "ACC", "gyr": "ANGVEL", "mag": "MAGN"}
SAMPLING_FREQUENCY_HZ = 100.0


def load_demographics(file_path: str | pathlib.Path) -> pd.DataFrame:
    """Load the demographics data.

    Parameters
    ----------
    file_path : str | pathlib.Path
        The path to the file containing the demographics data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing relevant demographics data.
    """
    # Parse file path
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    # Load demographics
    demographics_df = pd.read_csv(file_path, sep=",", header=0)

    # Adjust `id` column
    demographics_df["id"] = demographics_df["id"].apply(
        lambda s: "pp" + ("000" + str(s))[-3:]
    )
    demographics_df.rename(columns={"id": "sub_id"}, inplace=True)

    # Write out gender
    demographics_df["gender"] = demographics_df["gender"].map({0: "male", 1: "female"})
    return demographics_df


def load_marker_data(
    file_path: str | pathlib.Path,
    tracked_points: Optional[str | list[str]] = None,
    incl_err: bool = False,
    with_events: bool = False,
) -> SensorData | tuple[SensorData, pd.DataFrame]:
    """Load marker data from the given file path.

    Parameters
    ----------
    file_path : str | pathlib.Path
        The path to the data file, e.g., sub-<sub>_task-<task>[_run-<run>]_tracksys-omc_motion.tsv.
    tracked_points : Optional[str  |  list[str]], optional
        A list of tracked points (i.e., markers) for which to return the data, by default None.
        If None, data for all tracked points is returned.
    with_events : bool, optional
        Whether to return also annotated gait events, by default False

    Returns
    -------
    SensorData
        The sensor data for the given tracked points.
    """

    # Parse file path
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    # Set channel comps and types
    ch_comps = ["x", "y", "z"]
    ch_types = {"POS": "pos"}

    # Load channels
    channels = pd.read_csv(
        file_path.parent / file_path.name.replace("_motion.tsv", "_channels.tsv"),
        sep="\t",
        header=0,
    )
    sampling_frequency_Hz = channels["sampling_frequency"].iloc[0].astype(float)
    pos_units = channels[(channels["type"] == "POS")]["units"].iloc[0]

    # Load data
    data = pd.read_csv(file_path, sep="\t", header=0)
    if sampling_frequency_Hz != SAMPLING_FREQUENCY_HZ:
        data = data.iloc[
            :: int(sampling_frequency_Hz / SAMPLING_FREQUENCY_HZ)
        ].reset_index()

    # Create SensorData
    dataset = dict()
    for t in tracked_points:
        data_sel = data.loc[
            :,
            [
                f"{t}_{ch_type}_{ch_comp}"
                for ch_type in ch_types.keys()
                for ch_comp in ch_comps
            ],
        ]
        data_sel.columns = [
            f"{ch_types[ch_type]}_{ch_comp}"
            for ch_type in ch_types
            for ch_comp in ch_comps
        ]
        dataset[t] = data_sel

    # Convert units
    if pos_units == "mm":
        for _, sensor_data in dataset.items():
            sensor_data.loc[
                :,
                [f"{ch_type}_{ch_comp}" for ch_type in ["pos"] for ch_comp in ch_comps],
            ] /= 1000.0
    if not with_events:
        return dataset

    # Load events
    if with_events:
        events = pd.read_csv(
            file_path.parent
            / file_path.name.replace("_tracksys-omc_motion.tsv", "_events.tsv"),
            sep="\t",
            header=0,
        )
        if sampling_frequency_Hz != SAMPLING_FREQUENCY_HZ:
            events["onset"] = events["onset"] // (
                sampling_frequency_Hz / SAMPLING_FREQUENCY_HZ
            )
        return dataset, events


def load_imu_data(
    file_path: str | pathlib.Path,
    tracked_points: Optional[str | list[str]] = None,
    incl_magn: bool = False,
) -> SensorData:
    """Load IMU data from the given file path.

    Parameters
    ----------
    file_path : str | pathlib.Path
        The path to the data file, e.g., sub-<sub>_task-<task>[_run-<run>]_tracksys-imu_motion.tsv.
    tracked_points : Optional[str  |  list[str]], optional
        A list of tracked point for which to return the sensor data, by default None.
        If None, data for all tracked points is returned.
    incl_magn : bool, optional
        Whether to include the magnetometer data, by default False

    Returns
    -------
    SensorData
        The sensor data for the given tracked points.
    """
    # Parse file path
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    # Set channel comps and types
    ch_types = ["acc", "gyr", "mag"] if incl_magn else ["acc", "gyr"]
    ch_comps = ["x", "y", "z"]

    # Load channels
    channels = pd.read_csv(
        file_path.parent / file_path.name.replace("_motion.tsv", "_channels.tsv"),
        sep="\t",
        header=0,
    )
    sampling_frequency_Hz = channels["sampling_frequency"].iloc[0].astype(float)
    acc_units = channels[(channels["type"] == "ACC")]["units"].iloc[0]
    gyr_units = channels[(channels["type"] == "ANGVEL")]["units"].iloc[0]

    # Load sensor data
    data = pd.read_csv(file_path, sep="\t", header=0)
    if sampling_frequency_Hz != SAMPLING_FREQUENCY_HZ:
        data = data.iloc[
            :: int(sampling_frequency_Hz / SAMPLING_FREQUENCY_HZ)
        ].reset_index()

    # Create SensorData
    dataset = dict()
    for t in tracked_points:
        data_sel = data.loc[
            :,
            [
                f"{t}_{MAPPING_CHANNEL_TYPES_COLS[ch_type]}_{ch_comp}"
                for ch_type in ch_types
                for ch_comp in ch_comps
            ],
        ]
        data_sel.columns = [
            f"{ch_type}_{ch_comp}" for ch_type in ch_types for ch_comp in ch_comps
        ]
        dataset[t] = data_sel

    # Convert units
    if acc_units == "g":
        for _, sensor_data in dataset.items():
            sensor_data.loc[
                :,
                [f"{ch_type}_{ch_comp}" for ch_type in ["acc"] for ch_comp in ch_comps],
            ] *= 9.81
    if gyr_units == "rad/s":
        for _, sensor_data in dataset.items():
            sensor_data.loc[
                :,
                [f"{ch_type}_{ch_comp}" for ch_type in ["gyr"] for ch_comp in ch_comps],
            ] *= (
                180.0 / np.pi
            )
    return dataset
