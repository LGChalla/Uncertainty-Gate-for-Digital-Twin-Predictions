"""
Model constants, version 3. config.py and config_v2.py are unchanged.

One change from v2: fibroblasts sense their own lineage's density over a wider
disc (fib_sense_radius_um, 120 um) than the 60 um neighbourhood every other
local rule uses.

The why- With 60 um the disc holds 28 neighbours, so the logistic room
max(0, 1 - c / (28 K_fib)) cannot resolve a set point below about 1/28: one
neighbour already uses up a K_fib of 0.036. In the v2 campaign the fibroblast
share of cells never fell below ~0.12 however low K_fib went, while patients'
EPIC fibroblast share sits at 0.05-0.10 (coverage check on 8,614 v2 draws,
Nov 6: patients below the twin's range in every cohort). At 120 um the disc
holds 112 neighbours and the analytic steady state spans 0.6%-27% over the
new K_fib prior. A contact-inhibition sensing range of a few cell diameters
is a modelling choice (LC, Nov 6), not a measured value: status "find".
"""

from .config import P
from .config_v2 import TME_V2

TME_V3 = {
    **TME_V2,
    "fib_sense_radius_um": P(120.0, "um", "fibroblast density-sensing radius for contact inhibition; "
                             "chosen so the set point is resolvable (112 neighbours); LC Nov 6", "find"),
}
