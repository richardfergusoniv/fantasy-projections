"""Tests for rewrite go/no-go gate logic."""

from __future__ import annotations

from scripts.go_nogo_rewrite import decide


def test_blend_capturing_edge_means_do_not_rewrite():
    artifacts = {
        "market_edge": {
            "actionable_summary": {
                "v1": {"multi_season_edge": True},
                "v2": {"multi_season_edge": True},
                "blend": {"multi_season_edge": True},
                "carry_forward": {"multi_season_edge": True},
            },
            "seasons": {
                "2023": {
                    "models": {
                        "v1": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.30,
                            }
                        },
                        "v2": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.33,
                            }
                        },
                        "blend": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.34,
                            }
                        },
                        "carry_forward": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.20,
                            }
                        },
                    }
                },
                "2024": {
                    "models": {
                        "v1": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.40,
                            }
                        },
                        "v2": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.38,
                            }
                        },
                        "blend": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.42,
                            }
                        },
                        "carry_forward": {
                            "edge": {
                                "actionable_points_edge": True,
                                "edge_corr_residual_vs_actual_points": -0.15,
                            }
                        },
                    }
                },
            },
        },
        "ensemble": {
            "holdout_accuracy": {
                "v1": {"overall": {"points_mae": 34.0, "spearman": 0.74}},
                "v2": {"overall": {"points_mae": 32.5, "spearman": 0.73}},
                "blend": {"overall": {"points_mae": 31.7, "spearman": 0.76}},
            }
        },
        "rolling_eval": {"dispersion_summary": {"n_folds": 3}, "metadata": {}},
    }
    decision = decide(artifacts)
    assert decision["verdict"] == "do_not_rewrite"
    assert "blend" in decision["rationale"].lower()


def test_no_edge_means_do_not_rewrite():
    artifacts = {
        "market_edge": {
            "actionable_summary": {},
            "seasons": {
                "2025": {
                    "models": {
                        "v1": {
                            "edge": {
                                "actionable_points_edge": False,
                                "edge_corr_residual_vs_actual_points": 0.01,
                            }
                        },
                        "carry_forward": {
                            "edge": {
                                "actionable_points_edge": False,
                                "edge_corr_residual_vs_actual_points": 0.02,
                            }
                        },
                    }
                }
            },
        },
        "ensemble": {},
        "rolling_eval": {},
    }
    assert decide(artifacts)["verdict"] == "do_not_rewrite"
