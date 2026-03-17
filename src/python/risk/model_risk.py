"""Model risk quantification across different model specifications.

Computes model risk metrics by comparing VaR/ES estimates across
GBM, Heston, and Rough Heston models for the same portfolio.

Key metrics:
  - Model risk ratio: max(VaR) / min(VaR) across models
  - Model uncertainty spread: std of VaR across models
  - Relative model risk contribution per asset

TODO:
  - Implement model comparison framework
  - Regime-conditional model risk
"""


class ModelRiskAnalyser:
    """Compare risk estimates across different model specifications."""

    def __init__(self) -> None:
        self.results_: dict | None = None

    def compare(self, model_results: dict[str, dict]) -> dict:
        """Compare VaR/ES across models.

        Args:
            model_results: Dict mapping model_name -> {var_95, var_99, es_95, es_99}.

        Returns:
            Dict of model risk metrics.
        """
        # TODO: implement
        pass
