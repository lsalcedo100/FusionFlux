"""The confinement study, callable.

``pip install fusionflux`` installs exactly two things: this package and
``neutron_yield``. It deliberately does not install the analysis scripts. Those
are ``hdb5.py``, ``scaling_law.py``, ``dimensional.py`` and the twelve
``analysis_*.py`` modules, they are run from a checkout as ``python3 hdb5.py
train``, and every one of them needs a dataset that is fetched from OSF rather
than shipped. Installing them would put ``config``, ``storage``, ``validation``
and ``forecast`` on the import path of every environment this package lands in,
where they would shadow any other project's module of the same name, in exchange
for scripts that cannot run without a download the wheel does not provide.

So the installable surface is the part that works with nothing but the wheel: a
point estimate, a calibrated interval, an extrapolation distance, and a refusal,
all read from a few kilobytes of coefficients shipped inside the package.

    from fusionflux import predict

    result = predict(ip_ma=15.0, bt_t=5.3, ne_line_1e19_m3=10.0, p_loss_mw=87.0,
                     r_m=6.2, inverse_aspect_ratio=0.3226, kappa=1.7, m_eff_amu=2.5)

    result.tau_s                              # 2.837
    result.physics_exceeds_training_ceiling   # True
"""

from __future__ import annotations

from fusionflux.predictor import (
    ConfinementPrediction,
    ModelPrediction,
    ServiceCard,
    build_service_card,
    format_prediction,
    load_card,
    predict,
    save_card,
)

__all__ = [
    "ConfinementPrediction",
    "ModelPrediction",
    "ServiceCard",
    "build_service_card",
    "format_prediction",
    "load_card",
    "predict",
    "save_card",
]

__version__ = "0.2.1"
