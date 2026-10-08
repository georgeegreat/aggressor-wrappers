"""PASTA per-residue energy file → standard table.

PASTA 2.0 (Walsh et al., 2014, *Nucleic Acids Res* 42:W301) scores the best
pairing energy each residue can reach in a cross-beta arrangement, in its own
energy units. The scale is INVERTED relative to every other tool in this panel:
a more negative energy is more aggregation-prone, so the call is
``energy <= threshold``.

On the cutoff. The published operating point for peptide pairing is about -5.0
energy units, and that is what ``PREDICTOR_REGISTRY`` already declares as
``default_threshold``. This parser now reads that declaration instead of
carrying a second, divergent default of its own -- which is what
``FoldAmyloidParser`` does, and it keeps the published value written down once.

The cutoff matters more here than anywhere else in the panel, because PASTA's
energy distribution is dense near zero. Measured on a 227,106-residue
opisthokont set:

    <= -2.0   97.1 %      <= -4.0   53.1 %
    <= -2.8   87.2 %      <= -5.0   27.5 %   (published)

At -2.8 PASTA called seven residues in eight. A member that calls almost
everything is not a voter but a constant: it appeared in 100 % of the 2,654
consensus regions of the FXR1 orthologue sweep, raising the apparent support of
every one of them by exactly one tool and inflating tier assignment across the
board. At -5.0 it sits with APPNN (27.5 %) and Cross-Beta (25.1 %).

Note also that the runner's ``amount`` is the PASTA *server's* submission
parameter, deciding which pairings the service reports at all. It is not a
per-residue APR cutoff, and the two are deliberately kept apart: changing how
much the server reports should not silently change what counts as an APR.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

#: Fallback used only if the registry ever loses its declaration.
#: Walsh et al. (2014), the pairing-energy operating point.
PUBLISHED_ENERGY_THRESHOLD = -5.0


class PASTAParser(BasePredictorParser):
    spec = get_predictor_spec("pasta")

    def __init__(self, energy_threshold: float | None = None) -> None:
        self.energy_threshold = (
            float(energy_threshold)
            if energy_threshold is not None
            else float(
                self.spec.default_threshold
                if self.spec.default_threshold is not None
                else PUBLISHED_ENERGY_THRESHOLD
            )
        )

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        values = pd.read_csv(source, header=None, names=["energy"])
        if len(values) != len(sequence):
            raise ValueError(
                f"PASTA length mismatch: {len(values)} energies vs {len(sequence)} residues"
            )

        energies = [float(x) for x in values["energy"]]
        binary = binary_from_scores(
            energies,
            threshold=self.energy_threshold,
            greater_or_equal=False,
        )

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=energies,
            binary=binary,
            metadata={"energy_threshold": self.energy_threshold},
        )
