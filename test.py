import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib


ROOT = pathlib.Path("Z:\\Keep Control\\Data\\lab dataset\\rawdata")


def main() -> None:
    sub_id = "pp132"
    task = "walkPreferred"
    run = None

    marker_data = pd.read_csv(
        ROOT.joinpath(
            f"sub-{sub_id}",
            "motion",
            f"sub-{sub_id}_task-{task}_tracksys-omc_motion.tsv"
        ),
        sep="\t", 
        header=0
    )

    imu_data = pd.read_csv(
        ROOT.joinpath(
            f"sub-{sub_id}",
            "motion",
            f"sub-{sub_id}_task-{task}_tracksys-imu_motion.tsv"
        ),
        sep="\t", 
        header=0
    )

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(marker_data[["l_ank_POS_z", "r_ank_POS_z"]])
    axs[1].plot(imu_data[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]])
    for ax in axs:
        ax.grid(which="both", axis="both", alpha=0.2)
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    main()