"""Walk-forward VaR/ES backtesting engine.

Runs a strictly walk-forward backtest: at each test date, calibrates
models on all data up to t-1, simulates 1-day-ahead returns via the
C++ engine, computes VaR/ES, and records exceedances.
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', 'src', 'cpp'))

from calibration.quick_calibrator import (
    calibrate_gbm, calibrate_heston, calibrate_rough_heston, update_v0
)


def _load_engine():
    """Load C++ engine module."""
    import quant_engine_py as qe
    return qe


@dataclass
class DayResult:
    date: object
    model: str
    alpha: float
    var: float
    es: float
    actual_return: float
    exceedance: bool


@dataclass
class BacktestResults:
    day_results: list = field(default_factory=list)
    calibration_log: list = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {'Date': r.date, 'Model': r.model, 'Alpha': r.alpha,
             'VaR': r.var, 'ES': r.es, 'Actual': r.actual_return,
             'Exceedance': r.exceedance}
            for r in self.day_results
        ])

    def exceedance_series(self, model: str, alpha: float) -> np.ndarray:
        return np.array([
            r.exceedance for r in self.day_results
            if r.model == model and r.alpha == alpha
        ], dtype=bool)

    def var_series(self, model: str, alpha: float) -> np.ndarray:
        return np.array([
            r.var for r in self.day_results
            if r.model == model and r.alpha == alpha
        ])

    def es_series(self, model: str, alpha: float) -> np.ndarray:
        return np.array([
            r.es for r in self.day_results
            if r.model == model and r.alpha == alpha
        ])

    def actual_series(self, model: str, alpha: float) -> np.ndarray:
        return np.array([
            r.actual_return for r in self.day_results
            if r.model == model and r.alpha == alpha
        ])

    def dates(self, model: str, alpha: float) -> list:
        return [
            r.date for r in self.day_results
            if r.model == model and r.alpha == alpha
        ]


class VaRBacktester:
    """Walk-forward VaR/ES backtester using C++ Monte Carlo engine."""

    def __init__(self, returns: pd.Series, test_start_date: str,
                 models=None, recalibrate_every: int = 60,
                 n_paths: int = 50_000,
                 confidence_levels=None,
                 n_steps_per_day: int = 4):
        self.returns = returns
        self.test_start = pd.Timestamp(test_start_date)
        self.models = models or ['gbm', 'heston', 'rough_heston']
        self.recalibrate_every = recalibrate_every
        self.n_paths = n_paths
        self.alphas = confidence_levels or [0.95, 0.99]
        self.n_steps = n_steps_per_day
        self.qe = _load_engine()
        self.results = BacktestResults()

    def run(self) -> BacktestResults:
        dates = self.returns.index
        test_mask = dates >= self.test_start
        test_dates = dates[test_mask]
        n_test = len(test_dates)

        if n_test == 0:
            print("  No test dates found!")
            return self.results

        print(f"  Backtest: {n_test} test days, {len(self.models)} models, "
              f"{self.n_paths} paths")

        params = {}
        last_calib = -999
        calib_count = 0

        for idx, t in enumerate(test_dates):
            t_loc = dates.get_loc(t)
            train_returns = self.returns.iloc[:t_loc].values
            actual_ret = self.returns.iloc[t_loc]

            if len(train_returns) < 60:
                continue

            # Recalibrate if needed
            days_since = idx - last_calib
            if days_since >= self.recalibrate_every or last_calib < 0:
                try:
                    params = self._calibrate_all(train_returns)
                    last_calib = idx
                    calib_count += 1
                except Exception as e:
                    if not params:
                        print(f"  WARNING: calibration failed at {t}: {e}")
                        continue
            else:
                # Fast update: just refresh v0
                recent = train_returns[-80:]
                for model_name in ['heston', 'rough_heston']:
                    if model_name in params:
                        params[model_name] = self._update_v0(
                            params[model_name], recent, model_name
                        )

            # Simulate each model
            for model_name in self.models:
                if model_name == 'arima':
                    continue  # handled separately

                try:
                    sim_returns = self._simulate_day(
                        model_name, params.get(model_name), train_returns
                    )
                except Exception as e:
                    if idx % 200 == 0:
                        print(f"  WARNING: {model_name} sim failed at {t}: {e}")
                    continue

                for alpha in self.alphas:
                    q = (1 - alpha) * 100
                    var = -np.percentile(sim_returns, q)
                    tail = sim_returns[sim_returns <= -var]
                    es = -np.mean(tail) if len(tail) > 0 else var

                    exceedance = actual_ret < -var

                    self.results.day_results.append(DayResult(
                        date=t, model=model_name, alpha=alpha,
                        var=var, es=es, actual_return=actual_ret,
                        exceedance=exceedance
                    ))

            if (idx + 1) % 50 == 0 or idx == n_test - 1:
                pct = (idx + 1) / n_test * 100
                # Count exceedances so far for VaR99
                exc_99 = {}
                for m in self.models:
                    if m == 'arima':
                        continue
                    exc = self.results.exceedance_series(m, 0.99)
                    exc_99[m] = int(np.sum(exc)) if len(exc) > 0 else 0
                exc_str = ", ".join(f"{m}:{v}" for m, v in exc_99.items())
                print(f"  Day {idx+1}/{n_test} ({pct:.0f}%) | "
                      f"Calibrations: {calib_count} | "
                      f"VaR99 exc: {exc_str}")

        return self.results

    def _calibrate_all(self, train_returns: np.ndarray) -> dict:
        params = {}
        if 'gbm' in self.models:
            cal = calibrate_gbm(train_returns)
            params['gbm'] = cal
        if 'heston' in self.models:
            cal = calibrate_heston(train_returns)
            params['heston'] = cal
        if 'rough_heston' in self.models:
            cal = calibrate_rough_heston(train_returns)
            params['rough_heston'] = cal
        self.results.calibration_log.append(params.copy())
        return params

    def _update_v0(self, cal_params, recent_returns, model_name):
        """Fast v0 update without full recalibration."""
        if len(recent_returns) >= 60:
            rv_20 = np.var(recent_returns[-20:], ddof=1) * 252
            rv_60 = np.var(recent_returns[-60:], ddof=1) * 252
            v0 = max(rv_20, 0.7 * rv_60)
        else:
            v0 = np.var(recent_returns[-20:], ddof=1) * 252
        # Floor at 50% of long-run variance
        v0 = max(v0, 0.5 * cal_params.theta)
        if model_name == 'heston':
            from calibration.quick_calibrator import CalibratedHeston
            return CalibratedHeston(
                S0=cal_params.S0, mu=cal_params.mu, v0=v0,
                kappa=cal_params.kappa, theta=cal_params.theta,
                sigma_v=cal_params.sigma_v, rho=cal_params.rho
            )
        else:
            from calibration.quick_calibrator import CalibratedRoughHeston
            return CalibratedRoughHeston(
                S0=cal_params.S0, mu=cal_params.mu, v0=v0,
                kappa=cal_params.kappa, theta=cal_params.theta,
                sigma_v=cal_params.sigma_v, rho=cal_params.rho,
                H=cal_params.H
            )

    def _simulate_day(self, model_name: str, cal_params,
                      train_returns: np.ndarray) -> np.ndarray:
        """Simulate 1-day-ahead returns via C++ engine."""
        qe = self.qe

        config = qe.SimConfig()
        config.n_paths = self.n_paths
        config.n_steps = self.n_steps
        config.seed = 42
        config.antithetic = True

        S0 = 1.0  # normalize

        if model_name == 'gbm':
            p = qe.GBMParams()
            p.S0 = S0
            p.mu = cal_params.mu
            p.sigma = cal_params.sigma
            p.T = 1.0 / 252.0
            p.n_steps = self.n_steps
            terminal = np.array(qe.simulate_gbm(p, config))

        elif model_name == 'heston':
            p = qe.HestonParams()
            p.S0 = S0
            p.mu = cal_params.mu
            p.v0 = cal_params.v0
            p.kappa = cal_params.kappa
            p.theta = cal_params.theta
            p.sigma_v = cal_params.sigma_v
            p.rho = cal_params.rho
            p.T = 1.0 / 252.0
            p.n_steps = self.n_steps
            terminal = np.array(qe.simulate_heston(p, config, True))

        elif model_name == 'rough_heston':
            p = qe.RoughHestonParams()
            p.S0 = S0
            p.mu = cal_params.mu
            p.v0 = cal_params.v0
            p.kappa = cal_params.kappa
            p.theta = cal_params.theta
            p.sigma_v = cal_params.sigma_v
            p.rho = cal_params.rho
            p.H = cal_params.H
            p.T = 1.0 / 252.0
            p.n_steps = self.n_steps
            terminal = np.array(qe.simulate_rough_heston(p, config, 8))

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Convert terminal prices to 1-day returns
        sim_returns = terminal / S0 - 1.0
        return sim_returns


def add_arima_results(results: BacktestResults,
                      arima_forecasts: pd.DataFrame,
                      test_dates: pd.DatetimeIndex,
                      actual_returns: pd.Series):
    """Add ARIMA prediction interval exceedances to backtest results."""
    arima_dates = pd.to_datetime(arima_forecasts['date'])
    arima_forecasts = arima_forecasts.set_index(arima_dates)

    for t in test_dates:
        if t not in arima_forecasts.index:
            continue
        row = arima_forecasts.loc[t]
        actual = actual_returns.loc[t] if t in actual_returns.index else np.nan
        if np.isnan(actual):
            continue

        # 95% interval
        lower_95 = row['lower_95']
        var_95 = -lower_95  # VaR = -lower bound
        es_95 = var_95 * 1.1  # approximate ES from interval
        exc_95 = actual < lower_95

        results.day_results.append(DayResult(
            date=t, model='arima', alpha=0.95,
            var=var_95, es=es_95, actual_return=actual,
            exceedance=exc_95
        ))

        # 99% interval — approximate from 95% by scaling
        # ARIMA only provides 80% and 95%; scale for 99%
        lower_80 = row['lower_80']
        width_95 = row['predicted'] - lower_95
        width_99 = width_95 * 2.576 / 1.96  # normal scaling
        lower_99 = row['predicted'] - width_99
        var_99 = -lower_99
        es_99 = var_99 * 1.1
        exc_99 = actual < lower_99

        results.day_results.append(DayResult(
            date=t, model='arima', alpha=0.99,
            var=var_99, es=es_99, actual_return=actual,
            exceedance=exc_99
        ))
