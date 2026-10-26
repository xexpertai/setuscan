"""EXAMPLE severity policy.

The width bands below are illustrative values for a demo. They are NOT taken from a
design code and are NOT a structural certification. An asset owner's engineer must
set the bands (and the growth rule) for their structure type and exposure class;
`SeverityPolicy` is plain data so it can be loaded from configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Band:
    name: str
    max_mm: float  # upper edge (exclusive); the last band uses inf
    action: str  # "monitor" | "review" | "urgent_review"
    reinspect_days: int


@dataclass
class SeverityPolicy:
    label: str = "EXAMPLE policy (illustrative thresholds, not a code-certified standard)"
    bands: list[Band] = field(default_factory=lambda: [
        Band("hairline", 0.3, "monitor", 365),
        Band("fine", 1.0, "monitor", 180),
        Band("moderate", 2.0, "review", 30),
        Band("wide", float("inf"), "urgent_review", 7),
    ])
    # a width this close to a band edge is "ambiguous" and worth a closer look
    edge_margin_frac: float = 0.15
    # any significant growth between visits needs an engineer
    escalate_on_growth: bool = True

    def band_for(self, width_mm: float) -> Band:
        for b in self.bands:
            if width_mm < b.max_mm:
                return b
        return self.bands[-1]

    def edges(self) -> list[float]:
        return [b.max_mm for b in self.bands if b.max_mm != float("inf")]

    def nearest_edge(self, width_mm: float) -> tuple[float, float]:
        """(edge, relative distance) of the band edge nearest to width_mm."""
        e = min(self.edges(), key=lambda x: abs(x - width_mm))
        return e, abs(width_mm - e) / e

    def is_ambiguous(self, width_mm: float, mm_per_px: float) -> bool:
        e, rel = self.nearest_edge(width_mm)
        # ambiguous if within the margin, or within 1 working pixel of the edge
        return rel <= self.edge_margin_frac or abs(width_mm - e) <= mm_per_px

    def needs_engineer(self, width_mm: float, significant_growth: bool) -> bool:
        return self.band_for(width_mm).action != "monitor" or (self.escalate_on_growth and significant_growth)

    def to_dict(self) -> dict:
        d = asdict(self)
        for b in d["bands"]:
            if b["max_mm"] == float("inf"):
                b["max_mm"] = None
        return d
