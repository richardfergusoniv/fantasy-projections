"""Rolling backtest, calibration, and baseline reference models.

Import submodules directly (for example ``evaluation.accuracy_first``) so
serverless bundles do not eagerly load sklearn-backed baselines at startup.
"""
