from gauge_reader.eval.interval_hit import hit_rate, interval_hit

DUAL = [
    {"units": ["bar"], "interval": [3.0, 3.1], "point": False},
    {"units": ["psi"], "interval": [43.0, 45.0], "point": False},
]


def test_dual_scale_either_scale_is_a_hit():
    assert interval_hit(3.05, "bar", DUAL)
    assert interval_hit(44.0, "psi", DUAL)
    assert interval_hit(44.0, "PSI", DUAL)


def test_dual_scale_value_checked_against_its_own_unit():
    assert not interval_hit(44.0, "bar", DUAL)  # right number, wrong scale
    assert not interval_hit(3.05, "psi", DUAL)


def test_unknown_or_missing_unit_tries_all_scales():
    assert interval_hit(44.0, None, DUAL)
    assert interval_hit(3.05, "kPa", DUAL)


def test_point_answer_tolerance():
    alts = [{"units": ["°C"], "interval": [80.0, 80.0], "point": True}]
    assert not interval_hit(81.0, "°C", alts)
    assert interval_hit(81.0, "°C", alts, point_tol=0.02)


def test_json_string_and_hit_rate():
    import json

    s = json.dumps(DUAL)
    assert hit_rate([(3.05, "bar", s), (44.0, "bar", s)]) == 0.5


def test_hit_rate_by_group_separates_point_tol():
    from gauge_reader.eval.interval_hit import hit_rate_by_group

    pt = [{"units": ["psi"], "interval": [2100.0, 2100.0], "point": True}]
    rows = [("interval", 3.05, "bar", DUAL), ("interval", 9.0, "bar", DUAL), ("point_tol", 2100.0, "psi", pt)]
    assert hit_rate_by_group(rows) == {"interval": (0.5, 2), "point_tol": (1.0, 1)}


def test_reversed_interval_still_hits():
    # Some MeasureBench labels list the interval high-to-low.
    alts = [{"units": ["°C"], "interval": [32.0, 30.0], "point": False}]
    assert interval_hit(31.0, "°C", alts)
    assert not interval_hit(33.0, "°C", alts)
