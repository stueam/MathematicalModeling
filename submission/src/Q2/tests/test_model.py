import csv
import json
import math

import numpy as np
import pytest
from shapely.geometry import MultiPoint

from model_core import Design, metrics
from run import main


def test_enclosing_radius_is_not_half_the_diameter():
    triangle = MultiPoint([(0, 0), (2, 0), (1, math.sqrt(3))]).convex_hull
    radius, diameter = metrics(triangle)
    assert radius == pytest.approx(2 / math.sqrt(3))
    assert diameter == pytest.approx(2)
    assert radius > diameter / 2


def test_paper_point_symmetry_and_repeat_measurement():
    model = Design(p1=(0.0, 0.0), theta=0.0, reception=1200.0, nr=80, na=16, arc=128)
    positive = model.evaluate((840.0, 495.0), 0.125)
    negative = model.evaluate((840.0, -495.0), 0.125)
    assert positive['J'] == pytest.approx(23.147696339288032, abs=1e-8)
    assert negative['J'] == pytest.approx(positive['J'], abs=1e-8)
    assert positive['probability_sum'] == pytest.approx(1.0, abs=1e-12)
    assert model.evaluate(model.p1)['J'] == model.initial_radius
    assert model.area_partition_relative_error < 1e-5


def test_search_writes_complete_grid_refinement_and_summary(tmp_path, capsys):
    main(['--scenario', 'asymmetric', '--step', '1800', '--output', str(tmp_path)])
    summary = json.loads((tmp_path / 'summary.json').read_text())
    rows = []
    for filename in ('grid.csv', 'refined.csv'):
        with (tmp_path / filename).open(encoding='utf-8-sig', newline='') as handle:
            part = list(csv.DictReader(handle))
        assert part
        rows.extend(part)
    objectives = np.array([float(row['J']) for row in rows])
    assert np.isfinite(objectives).all()
    assert summary['best']['J'] == min(objectives)
    assert all(float(row['probability_sum']) == pytest.approx(1.0) for row in rows)
    assert json.loads(capsys.readouterr().out)['best'] == summary['best']


@pytest.mark.parametrize(
    'arguments',
    [
        ['--step', '0'],
        ['--step', 'nan'],
        ['--step', 'inf'],
        ['--step', '4000'],
        ['--R0', '999'],
        ['--R0', 'nan'],
        ['--evaluate', 'nan', '0'],
        ['--evaluate', 'inf', '0'],
        ['--evaluate', '1801', '0'],
    ],
)
def test_invalid_numeric_inputs_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
