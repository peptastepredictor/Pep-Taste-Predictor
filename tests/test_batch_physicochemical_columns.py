from main import build_batch_physicochemical_row


def test_build_batch_physicochemical_row_returns_expected_keys():
    row = build_batch_physicochemical_row("ACDE")

    assert "Length" in row
    assert "Molecular Weight (Da)" in row
    assert "Helix Fraction" in row
    assert "Hydrophobic (%)" in row
    assert row["Length"] == 4
