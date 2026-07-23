"""Static constants for the current B0531+21 RFI baseline."""

from __future__ import annotations

import numpy as np

B0531_FILE_KEY = "B0531+21_59000_48386.fil"

N_CHANNELS = 256
N_TIME = 256

FREQUENCY_RANGES = {
    B0531_FILE_KEY: np.linspace(1530.0, 1210.0, N_CHANNELS),
}

RFI_BLOCKS = {
    B0531_FILE_KEY: {
        1: (204, 1527),
        2: (2023, 4436),
        3: (4437, 4953),
        4: (6701, 7631),
        5: (42546, 45015),
        6: (45622, 45705),
        7: (46186, 46927),
        8: (46928, 47303),
        9: (48686, 48831),
        10: (48832, 49591),
    }
}

SEGMENT_RFI_CHANNELS = {
    B0531_FILE_KEY: {
        1: [[230, 235], [239, 241]],
        2: [230, 256],
        3: [230, 256],
        4: [[230, 235], [239, 243]],
        5: [[189, 195], [199, 202], [205, 209]],
        6: [[189, 195], [199, 202], [206, 210]],
        7: [[190, 193], [199, 202], [206, 210]],
        8: [[188, 194], [198, 203], [206, 210]],
        9: [[188, 194], [198, 203], [206, 210]],
        10: [[188, 194], [198, 203], [206, 210]],
    }
}

FEATURE_COLUMNS_ORIGINAL = [
    "mean_o",
    "std_o",
    "skew_o",
    "kurt_o",
    "mean_o_ratio",
    "std_o_ratio",
    "skew_o_ratio",
    "kurt_o_ratio",
]

FEATURE_COLUMNS_NORMALIZED = [
    "mean_n",
    "std_n",
    "skew_n",
    "kurt_n",
    "mean_n_ratio",
    "std_n_ratio",
    "skew_n_ratio",
    "kurt_n_ratio",
]

FEATURE_COLUMNS = [
    "mean_o",
    "std_o",
    "skew_o",
    "kurt_o",
    "mean_n",
    "std_n",
    "skew_n",
    "kurt_n",
    "mean_o_ratio",
    "std_o_ratio",
    "skew_o_ratio",
    "kurt_o_ratio",
    "mean_n_ratio",
    "std_n_ratio",
    "skew_n_ratio",
    "kurt_n_ratio",
]

FEATURE_SETS = {
    "orig": FEATURE_COLUMNS_ORIGINAL,
    "norm": FEATURE_COLUMNS_NORMALIZED,
    "both": FEATURE_COLUMNS,
}
