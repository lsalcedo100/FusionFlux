from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = DATA_PROCESSED_DIR / "plots"
MODELS_DIR = DATA_PROCESSED_DIR / "models"

RANDOM_STATE = 42
LAWSON_DT_IGNITION = 3e21
HIGH_YIELD_PERCENTILE = 0.90
LOW_LAWSON_RATIO_THRESHOLD = 0.50
SYNTHETIC_DATASET_ROWS = 600
HOLDOUT_TEST_SIZE = 0.2
MIN_TOTAL_SAMPLES = 15
MIN_TRAIN_SAMPLES = 12
MIN_TEST_SAMPLES = 3
MIN_GROUPED_HOLDOUT_GROUPS = 5
MIN_CV_FOLDS = 3
MAX_CV_FOLDS = 5

DATASET_FILENAME_CANDIDATES = (
    "nuclear_fusion_experiment.csv",
    "nuclear_fusion_data.csv",
    "fusion_dataset.csv",
    "synthetic_nuclear_fusion_experiment.csv",
)

COLUMN_ALIASES = {
    "shot_id": ["shot_id", "shot", "experiment_id", "pulse_id", "pulse", "run_id"],
    "fuel_density_m3": [
        "fuel_density_m3",
        "fuel_density",
        "density_m3",
        "density",
        "particle_density",
        "ion_density",
    ],
    "temperature_keV": [
        "temperature_keV",
        "temperature_kev",
        "ion_temperature_keV",
        "temp_kev",
    ],
    "temperature_eV": [
        "temperature_eV",
        "temperature_ev",
        "ion_temperature_eV",
        "temp_ev",
    ],
    "temperature_K": [
        "temperature_K",
        "temperature_k",
        "temperature_kelvin",
        "ion_temperature_K",
        "temp_K",
    ],
    "temperature": ["temperature", "ion_temperature", "plasma_temperature", "temp"],
    "temperature_unit": ["temperature_unit", "temp_unit", "unit_temperature"],
    "confinement_time_s": [
        "confinement_time_s",
        "confinement_time",
        "tau_E",
        "tau_e",
        "energy_confinement_time_s",
        "tau",
    ],
    "fuel_purity": ["fuel_purity", "purity", "fuel_mix_purity", "dt_purity"],
    "energy_input_MJ": [
        "energy_input_MJ",
        "energy_input",
        "input_energy_MJ",
        "heating_energy_MJ",
    ],
    "pressure_Pa": ["pressure_Pa", "pressure", "plasma_pressure_Pa", "chamber_pressure_Pa"],
    "neutron_yield": [
        "neutron_yield",
        "neutron_yield_per_second",
        "yield",
        "fusion_yield",
    ],
    "power_output_MW": ["power_output_MW", "power_output", "fusion_power_MW"],
    "Ip_MA": ["Ip_MA", "plasma_current_MA", "ip_ma", "current_MA"],
    "Bt_T": ["Bt_T", "magnetic_field_T", "bt_t", "field_T"],
    "R_m": ["R_m", "major_radius_m", "major_radius"],
    "a_m": ["a_m", "minor_radius_m", "minor_radius"],
    "kappa": ["kappa", "elongation"],
    "ne_20": ["ne_20", "electron_density_1e20_m3", "electron_density_norm"],
    "M_amu": ["M_amu", "ion_mass_amu", "fuel_mass_amu"],
    "Pin_MW": ["Pin_MW", "input_power_MW", "heating_power_MW", "power_input_MW"],
    "tau_E_measured_s": [
        "tau_E_measured_s",
        "tau_e_measured",
        "measured_confinement_time_s",
    ],
    "time_s": ["time_s", "time", "time_seconds"],
    "time_ms": ["time_ms", "time_milliseconds"],
}

BASE_FEATURE_COLUMNS = [
    "fuel_density_m3",
    "temperature_keV",
    "confinement_time_s",
    "fuel_purity",
    "energy_input_MJ",
    "pressure_Pa",
    "Ip_MA",
    "Bt_T",
    "R_m",
    "a_m",
    "kappa",
    "ne_20",
    "M_amu",
    "Pin_MW",
]

ENGINEERED_FEATURE_COLUMNS = [
    "triple_product",
    "lawson_ratio",
    "density_temp",
    "density_tau",
    "purity_weighted_density",
    "tau_E_ipb98_s",
    "log_fuel_density_m3",
    "log_temperature_keV",
    "log_confinement_time_s",
    "log_energy_input_MJ",
    "log_pressure_Pa",
    "log_triple_product",
]

TARGET_COLUMN = "neutron_yield"
TARGET_LOG_COLUMN = "log_neutron_yield"
GROUP_COLUMN = "shot_id"

LEAKAGE_COLUMNS = [
    "power_output_MW",
    "power_output",
    "fusion_gain_Q",
    "Q",
    "total_energy_out",
    "final_energy_MJ",
    "disruption_flag",
    "disruption_indicator",
]
