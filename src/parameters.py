# ---------- ATC constants ----------
DAYS_ATC = ["0109", "0217", "0424", "0505", "0508"]
SOCIAL_RELATIONS_JP = ["idontknow", "koibito", "doryo", "kazoku", "yuujin"]
SOCIAL_RELATIONS_EN = ["idontknow", "Couples", "Colleagues", "Families", "Friends"]
BOUNDARIES_ATC = {"xmin": -41000, "xmax": 49000, "ymin": -27500, "ymax": 24000}
BOUNDARIES_ATC_CORRIDOR = {"xmin": 5000, "xmax": 48000, "ymin": -27000, "ymax": 8000}


# ---------- DIAMOR constants ----------
DAYS_DIAMOR = ["06", "08"]
INTENSITIES_OF_INTERACTION_NUM = ["0", "1", "2", "3"]
BOUNDARIES_DIAMOR = {
    "06": {"xmin": -150, "xmax": 60000, "ymin": -5300, "ymax": 11100},
    "08": {"xmin": 170, "xmax": 60300, "ymin": -3700, "ymax": 8300},
}


DAYS = {"atc": DAYS_ATC, "diamor": DAYS_DIAMOR}


# ---------- trajectories ----------
MIN_VEL = 500  # m/s
MAX_VEL = 3000  # m/s
MIN_DURATION = 10  # s


# ---------- sources and sinks ----------
# DIAMOR
# SOURCES_DIAMOR = [
#     {"name": "N1", "xmin": 14000, "xmax": 18300, "ymin": 9400, "ymax": 11500},
#     {"name": "S1", "xmin": 8200, "xmax": 12500, "ymin": -5300, "ymax": -3800},
#     {"name": "N2", "xmin": 6700, "xmax": 10000, "ymin": 7100, "ymax": 8500},
#     {"name": "S2", "xmin": 5500, "xmax": 8500, "ymin": -3500, "ymax": -2300},
#     {"name": "W", "xmin": 200, "xmax": 3500, "ymin": -2000, "ymax": 6000},
#     {"name": "E", "xmin": 55000, "xmax": 60300, "ymin": -2000, "ymax": 6000},
# ]

SOURCES_DIAMOR = {
    "06": {
        "E": {"xmin": 58000, "xmax": 60000, "ymin": 0, "ymax": 6000},
        "W": {"xmin": 0, "xmax": 3000, "ymin": -500, "ymax": 5000},
        "NW": {"xmin": 3000, "xmax": 5500, "ymin": 5000, "ymax": 8500},
        "N": {"xmin": 13500, "xmax": 18000, "ymin": 10000, "ymax": 11500},
        "S": {"xmin": 8000, "xmax": 13000, "ymin": -6500, "ymax": -4000},
    },
    #     "06": {
    #     "E": {"xmin": 58000, "xmax": 60000, "ymin": -1000, "ymax": 7000},
    #     "W": {"xmin": 0, "xmax": 4000, "ymin": -500, "ymax": 4500},
    #     "NW": {"xmin": 3000, "xmax": 5500, "ymin": 5000, "ymax": 8500},
    #     "N": {"xmin": 13500, "xmax": 19000, "ymin": 10500, "ymax": 11500},
    #     "S": {"xmin": 8000, "xmax": 13000, "ymin": -6500, "ymax": -4000},
    # },
    "08": {
        "E": {"xmin": 58000, "xmax": 60300, "ymin": -2300, "ymax": 5900},
        "W": {"xmin": 0, "xmax": 2000, "ymin": -2300, "ymax": 5900},
        "N1": {"xmin": 6100, "xmax": 11200, "ymin": 7500, "ymax": 8500},
        "S1": {"xmin": 6100, "xmax": 11200, "ymin": -3800, "ymax": -2600},
        # "N2": {"xmin": 13671, "xmax": 17171, "ymin": 6352, "ymax": 7352},
        # "S2": {"xmin": 12671, "xmax": 16171, "ymin": -3648, "ymax": -2448},
        # "N3": {"xmin": 27671, "xmax": 30171, "ymin": 5852, "ymax": 7352},
        # "S3": {"xmin": 27671, "xmax": 30171, "ymin": -3148, "ymax": -2148},
        # "N4": {"xmin": 33171, "xmax": 36171, "ymin": 5852, "ymax": 7352},
        # "S4": {"xmin": 32171, "xmax": 36171, "ymin": -3648, "ymax": -2648},
        # "S5": {"xmin": 43671, "xmax": 4671, "ymin": -2148, "ymax": -2348},
        # "S6": {"xmin": 51671, "xmax": 53671, "ymin": -3148, "ymax": -1648},
    },
}


TURN_TYPE_DIAMOR = {
    "06": {
        ("E", "W"): "straight",
        ("W", "E"): "straight",
        ("N", "S"): "straight",
        ("S", "N"): "straight",
        ("E", "N"): "right",
        ("N", "E"): "left",
        ("E", "S"): "left",
        ("S", "E"): "right",
        ("E", "NW"): "right",
        ("NW", "E"): "left",
        ("N", "NW"): "right",
        ("NW", "N"): "left",
        ("N", "W"): "right",
        ("W", "N"): "left",
        ("NW", "S"): "right",
        ("S", "NW"): "left",
        ("S", "W"): "left",
        ("W", "S"): "right",
        ("NW", "W"): "right",
        ("W", "NW"): "left",
    },
    "08": {
        ("E", "W"): "straight",
        ("W", "E"): "straight",
        ("N1", "S1"): "straight",
        ("S1", "N1"): "straight",
        ("E", "N1"): "right",
        ("N1", "E"): "left",
        ("E", "S1"): "left",
        ("S1", "E"): "right",
        ("S1", "W"): "left",
        ("W", "S1"): "right",
        ("N1", "W"): "right",
        ("W", "N1"): "left",
    },
}

STRAIGHT_PAIRS_DIAMOR = {
    "06": [
        ("E", "W"),
        ("W", "E"),
        ("N", "S"),
        ("S", "N"),
    ],
    "08": [
        ("E", "W"),
        ("W", "E"),
        ("N1", "S1"),
        ("S1", "N1"),
    ],
}

STRONG_TURN_PAIRS_DIAMOR = {
    "06": [
        ("E", "N"),
        ("E", "S"),
        ("S", "E"),
        ("N", "E"),
        ("N", "NW"),
        ("N", "W"),
        ("NW", "N"),
        ("NW", "S"),
        ("S", "NW"),
        ("S", "W"),
        ("W", "N"),
        ("W", "S"),
        ("E", "W"),
        ("W", "E"),
        ("N", "S"),
        ("S", "N"),
    ],
    "08": [
        ("E", "N1"),
        ("E", "S1"),
        ("S1", "E"),
        ("N1", "E"),
        ("S1", "W"),
        ("W", "N1"),
        ("W", "S1"),
        ("E", "W"),
        ("W", "E"),
        ("N1", "S1"),
        ("S1", "N1"),
    ],
}

SOURCES_ATC = {
    "W": {"xmin": -850, "xmax": 8200, "ymin": -3750, "ymax": 8400},
    "E": {"xmin": 34000, "xmax": 40500, "ymin": -23900, "ymax": -15800},
}


# ---------- space binning ----------
SIZE_BIN = 500  # mm

# ---------- filtering before meta trajectories ----------
MIN_N_TRAJECTORIES = 10
CELL_SIZE = 500  # mm
MIN_TRAJECTORIES_BAD = 1
RATIO_BAD_CELLS = 0.05


# ---------- curvature ----------
WINDOW_SIZE = 20  # window size for smoothing


# ---------- plotting ----------
N_BINS = 16
