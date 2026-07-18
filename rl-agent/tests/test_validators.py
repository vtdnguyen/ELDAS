"""Run the GĐ 2.1 correctness validators as part of the test suite.

  * ``validate_cmdp_kkt``         — KKT / zero-duality-gap / Pareto monotonicity
                                    of the dual mechanism (G1.5 math gate).
  * ``validate_lambda_vs_omnisafe`` — λ-update law vs the reference controller
                                    (G1.4). Falls back to the inline Stooke
                                    reference when OmniSafe is absent.

Both return exit code 0 on success; we assert that.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim import validate_cmdp_kkt, validate_lambda_vs_omnisafe


def test_cmdp_kkt_conditions_hold(capsys):
    rc = validate_cmdp_kkt.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "VERDICT: PASS" in out


def test_lambda_update_matches_reference(capsys):
    rc = validate_lambda_vs_omnisafe.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "VERDICT" in out
