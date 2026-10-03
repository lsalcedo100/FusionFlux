"""What the delivered STD5 file is, row by row, relative to the full DB5.2.3 file.

The database description gives the standard analysis set STD5 as 7537 points.
The file the OSF deposit delivers under that name has 6228 rows. This settles
what those rows are by matching every one of them to the full DB5.2.3 file on
(device, shot, time) and reading the full file's selection flag and phase.

Run ``python3 tools/audit_std5_membership.py`` to regenerate
``results/std5_membership.json``. It needs both ITPA files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hdb5  # noqa: E402
import replication  # noqa: E402

RESULTS = ROOT / "results"

# The deposit splits two devices by wall era; the full file labels the device.
WALL_ERA = {"JETILW": "JET", "AUGW": "AUG"}
# H-mode phases with an ELM classification. "H" is H-mode with none recorded.
ELMY = ("HGELM", "HGELMH", "HSELM", "HSELMH")


def audit() -> dict:
    std5 = pd.read_csv(hdb5.default_hdb5_path(), low_memory=False)
    full = replication.load_db523_raw()

    delivered = std5.assign(
        device=std5["TOK"].replace(WALL_ERA), key_time=std5["TIME"].round(4)
    )
    full = full.assign(key_time=full["TIME"].round(4))
    keys = set(zip(delivered["device"], delivered["SHOT"], delivered["key_time"], strict=True))
    full["delivered"] = [
        (t, s, k) in keys for t, s, k in zip(full["TOK"], full["SHOT"], full["key_time"], strict=True)
    ]
    position = {
        (t, s, k): i + 1
        for i, (t, s, k) in enumerate(zip(full["TOK"], full["SHOT"], full["key_time"], strict=True))
    }
    delivered["full_row"] = [
        position.get((t, s, k)) for t, s, k in zip(delivered["device"], delivered["SHOT"], delivered["key_time"], strict=True)
    ]

    flagged = full[full["SELDB5"] == 1]
    elmy = flagged[flagged["PHASE"].isin(ELMY)]
    omitted_elmy = elmy[~elmy["delivered"]]
    unmatched = int(delivered["full_row"].isna().sum())

    return {
        "std5_file_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
        "db523_file_sha256": replication.verify_db523_file(replication.default_db523_path()).sha256,
        "delivered_rows": int(len(delivered)),
        "delivered_rows_matched_in_full_file": int(len(delivered) - unmatched),
        "delivered_rows_unmatched": unmatched,
        # IND in the deposit is the 1-based row number of the full file.
        "ind_is_full_file_row_number": bool((delivered["IND"] == delivered["full_row"]).all()),
        "delivered_rows_with_seldb5": int(full.loc[full["delivered"], "SELDB5"].eq(1).sum()),
        "delivered_phase_counts": {
            k: int(v) for k, v in full.loc[full["delivered"], "PHASE"].value_counts().items()
        },
        "full_file_rows": int(len(full)),
        "seldb5_rows_in_full_file": int(len(flagged)),
        "seldb5_rows_by_phase": {k: int(v) for k, v in flagged["PHASE"].value_counts().items()},
        "seldb5_elmy_rows": int(len(elmy)),
        "seldb5_elmy_rows_delivered": int(elmy["delivered"].sum()),
        "seldb5_elmy_rows_omitted": int(len(omitted_elmy)),
        "seldb5_elmy_rows_omitted_by_device": {
            k: int(v) for k, v in omitted_elmy["TOK"].value_counts().items()
        },
        "seldb5_non_elmy_rows": int(len(flagged) - len(elmy)),
        "seldb5_non_elmy_rows_delivered": int(flagged.loc[~flagged["PHASE"].isin(ELMY), "delivered"].sum()),
        "seldb5_devices_absent_from_delivered_file": sorted(
            set(flagged["TOK"]) - set(delivered["device"])
        ),
        "omitted_elmy_rows": omitted_elmy[["TOK", "SHOT", "TIME", "DATE", "PHASE", "ELMTYPE"]]
        .assign(full_row=omitted_elmy.index + 1)
        .to_dict(orient="records"),
    }


def main() -> None:
    report = audit()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "std5_membership.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{report['delivered_rows']} delivered rows, {report['delivered_rows_matched_in_full_file']} matched, "
        f"{report['delivered_rows_with_seldb5']} flagged SELDB5; phases {report['delivered_phase_counts']}"
    )
    print(
        f"SELDB5 rows in the full file: {report['seldb5_rows_in_full_file']}, of which ELMy "
        f"{report['seldb5_elmy_rows']} ({report['seldb5_elmy_rows_delivered']} delivered, "
        f"{report['seldb5_elmy_rows_omitted']} omitted) and non-ELMy {report['seldb5_non_elmy_rows']} "
        f"({report['seldb5_non_elmy_rows_delivered']} delivered)"
    )
    print(f"SELDB5 devices absent from the delivered file: {report['seldb5_devices_absent_from_delivered_file']}")
    print(f"Wrote {RESULTS / 'std5_membership.json'}")


if __name__ == "__main__":
    main()
