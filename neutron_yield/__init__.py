"""Neutron-yield pipeline: ML infrastructure, not a source of physical claims.

This package predicts ``neutron_yield`` from plasma operating conditions and
ships with a *synthetic* demo dataset, so any accuracy number it produces
measures how learnable that generator is and nothing else. None of the results
in ``results/RESULTS.md`` come from it.

It is kept, and packaged separately from the modules that do carry results, for
the engineering: a versioned preprocessing contract, atomic run publishing, and
artifact compatibility enforced even when the inference API is bypassed. See
``docs/neutron-yield-pipeline.md``.

The scientific pipeline (``hdb5``, ``scaling_law``, ``analysis_*``) does not
import anything from here; it shares only ``config`` and ``storage``.
"""
