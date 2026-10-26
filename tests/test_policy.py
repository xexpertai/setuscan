from setuscan.policy import SeverityPolicy


def test_bands():
    p = SeverityPolicy()
    assert p.band_for(0.1).name == "hairline"
    assert p.band_for(0.3).name == "fine"
    assert p.band_for(1.5).name == "moderate"
    assert p.band_for(5.0).name == "wide"
    assert "EXAMPLE" in p.label


def test_engineer_needed():
    p = SeverityPolicy()
    assert not p.needs_engineer(0.5, significant_growth=False)
    assert p.needs_engineer(0.5, significant_growth=True)
    assert p.needs_engineer(1.2, significant_growth=False)


def test_ambiguity_near_edges():
    p = SeverityPolicy()
    assert p.is_ambiguous(0.95, mm_per_px=0.05)  # 5% under the 1.0 mm edge
    assert p.is_ambiguous(1.08, mm_per_px=0.05)
    assert not p.is_ambiguous(0.6, mm_per_px=0.05)
    assert p.is_ambiguous(0.6, mm_per_px=0.45)  # one coarse pixel spans the edge
    assert p.nearest_edge(1.7)[0] == 2.0


def test_policy_serialisable():
    d = SeverityPolicy().to_dict()
    assert d["bands"][-1]["max_mm"] is None
