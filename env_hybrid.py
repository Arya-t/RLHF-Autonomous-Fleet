import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    step_minutes: int = 5
    max_steps: int = 288  # 24h with 5-min step
    ttl_minutes: int = 30
    pricing_interval_steps: int = 24  # 2h
    rebalance_interval_steps: int = 12  # 60min
    # NOTE: keep action key name "price" and field name "cs_share_levels" for compatibility.
    # Semantics now: surge price multiplier levels.
    cs_share_levels: Tuple[float, ...] = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    fixed_cs_share: float = 0.85  # CS keeps 85%, platform keeps 15% on CS orders.
    base_fee: float = 3.0
    unit_fare_per_km: float = 2.0
    min_bill_km: float = 1.0
    av_mu: float = 0.40
    cs_mu: float = 0.30
    av_init_demand_ratio: float = 0.30
    cs_init_demand_ratio: float = 0.15
    av_min_units: float = 8.0
    cs_min_units: float = 4.0
    av_order_cost: float = 1.0
    cs_order_cost: float = 2.0  # platform-side overhead per CS order (excluding payout share)
    av_fixed_cost_per_unit_step: float = 0.08
    timeout_penalty: float = 12.0
    imbalance_penalty: float = 0.0
    workload_cap_floor: float = 0.5
    workload_clip_max: float = 10.0
    rebalance_cost_per_unit_km: float = 0.70
    rebalance_adj_cost_mult: float = 1.0
    rebalance_non_adj_cost_mult: float = 1.8
    rebalance_max_send_ratio: float = 0.20
    rebalance_allow_all_pairs: bool = True
    rebalance_max_travel_steps: Optional[int] = None
    demand_price_elasticity: float = 1.3
    cs_supply_price_elasticity: float = 0.4
    cs_supply_min_multiplier: float = 0.70
    cs_supply_max_multiplier: float = 1.30
    cs_recover_rate: float = 0.15
    demand_multiplier: float = 0.01
    # Optional hard-rule penalty (for ablation only): keep default disabled.
    rule_penalty_enable: bool = False
    price_change_penalty_coef: float = 0.0
    imbalance_penalty_coef: float = 0.0
    demand_period_factors: Tuple[float, ...] = (
        0.75, 0.70, 0.75, 0.90, 1.00, 1.25, 1.05, 1.00, 1.02, 1.15, 1.30, 1.00
    )
    random_seed: int = 20240908


class HybridFulfillmentEnv:
    """
    Hybrid AV + crowdsourced fulfillment simulator with:
    - multi-region OD demand
    - cross-region service delay
    - timeout queue by waiting-age bucket
    - slow pricing action + medium-frequency AV rebalance action
    """

    def __init__(self, dataset_path: str, config: EnvConfig = None):
        self.config = config or EnvConfig()
        with open(dataset_path, "rb") as f:
            self.allarea_set = pickle.load(f)

        first_t = min(self.allarea_set.keys())
        self.region_dict = self.allarea_set[first_t].region_dict
        self.region_ids = sorted(self.region_dict.keys())
        self.n = len(self.region_ids)
        self.id_to_idx = {rid: i for i, rid in enumerate(self.region_ids)}
        self.period_num = len(self.allarea_set)
        self.slot_len = max(1, self.config.max_steps // self.period_num)
        self.ttl_steps = max(1, int(math.ceil(self.config.ttl_minutes / self.config.step_minutes)))

        self.rng = np.random.default_rng(self.config.random_seed)

        # Static matrices
        self.travel_steps = np.ones((self.n, self.n), dtype=int)
        self.travel_km = np.zeros((self.n, self.n), dtype=float)
        self.adj_mask = np.zeros((self.n, self.n), dtype=bool)
        self.base_od = np.zeros((self.period_num, self.n, self.n), dtype=float)
        self.base_mass = np.zeros(self.n, dtype=float)
        self.base_mu = np.zeros(self.n, dtype=float)

        self._build_static_tables()
        self._enforce_reachable_od_by_ttl()
        self._build_rebalance_mask()
        self.reset()

    def _build_static_tables(self):
        for rid_i in self.region_ids:
            i = self.id_to_idx[rid_i]
            reg_i = self.region_dict[rid_i]
            self.base_mu[i] = float(getattr(reg_i, "mu", self.config.av_mu))
            self.base_mass[i] = float(np.nansum(reg_i.od_demand[0]))

            # adjacency for AV rebalance
            nbrs = set(getattr(reg_i, "adjacent_list", []))
            for rid_j in self.region_ids:
                j = self.id_to_idx[rid_j]
                if rid_i == rid_j:
                    self.adj_mask[i, j] = True
                if rid_j in nbrs:
                    self.adj_mask[i, j] = True

                tt = getattr(reg_i, "travel_time_step", {}).get(rid_j, 1)
                self.travel_steps[i, j] = max(1, int(tt))

                c1 = np.array(getattr(reg_i, "centroid_xy", (0.0, 0.0)), dtype=float)
                c2 = np.array(getattr(self.region_dict[rid_j], "centroid_xy", (0.0, 0.0)), dtype=float)
                self.travel_km[i, j] = float(np.linalg.norm(c1 - c2) / 1000.0)

            for t in range(self.period_num):
                # allarea_set[t] is compatible and contains od_demand over all slots
                reg_t = self.allarea_set[t].region_dict[rid_i]
                row = np.asarray(reg_t.od_demand[t], dtype=float)
                row = np.nan_to_num(row, nan=0.0)
                self.base_od[t, i, :] = row

    def _enforce_reachable_od_by_ttl(self):
        """
        Enforce that all OD pairs are physically reachable within TTL.
        This keeps timeout attribution focused on supply insufficiency.
        """
        max_step = max(1, self.ttl_steps - 1)
        self.travel_steps = np.clip(self.travel_steps, 1, max_step)

    def _build_rebalance_mask(self):
        # Rebalance can be adjacency-only or all-pairs; diagonal is always disallowed.
        if self.config.rebalance_allow_all_pairs:
            mask = np.ones((self.n, self.n), dtype=bool)
        else:
            mask = self.adj_mask.copy()
        np.fill_diagonal(mask, False)
        if self.config.rebalance_max_travel_steps is not None:
            max_steps = int(self.config.rebalance_max_travel_steps)
            mask &= (self.travel_steps <= max_steps)
        self.rebalance_mask = mask

    def _init_supply(self):
        # Initialize AV/CS units from demand scale.
        # One AV unit provides av_mu orders/step, one CS unit provides cs_mu orders/step.
        avg_orders_per_step = np.mean(self.base_od, axis=(0, 2))
        av_units = np.maximum(
            self.config.av_min_units,
            self.config.av_init_demand_ratio * avg_orders_per_step / max(self.config.av_mu, 1e-8),
        )
        cs_units = np.maximum(
            self.config.cs_min_units,
            self.config.cs_init_demand_ratio * avg_orders_per_step / max(self.config.cs_mu, 1e-8),
        )
        return av_units.astype(float), cs_units.astype(float), cs_units.astype(float)

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        # Backward-compatible key name; now represents surge price-multiplier index.
        self.price_idx = np.full(self.n, 1, dtype=int)
        self.last_price_idx = self.price_idx.copy()
        self.backlog = np.zeros((self.n, self.n, self.ttl_steps), dtype=float)  # origin, dest, wait_age
        self.av_units, self.cs_units, self.cs_base_units = self._init_supply()
        # Split AV in-transit into service vs rebalance so utilization can exclude rebalance.
        self.transit_av_service: List[Tuple[int, int, float]] = []   # (arrive_step, dest_idx, units)
        self.transit_av_rebalance: List[Tuple[int, int, float]] = [] # (arrive_step, dest_idx, units)
        self.transit_cs: List[Tuple[int, int, float]] = []
        self.metrics = {
            "reward_sum": 0.0,
            "timeout_sum": 0.0,
            "served_sum": 0.0,
            "rebalance_sum": 0.0,
            "rebalance_km_sum": 0.0,
            "empty_cost_sum": 0.0,
            "av_fixed_cost_sum": 0.0,
            "revenue_sum": 0.0,
            "demand_util_sum": 0.0,
            "demand_util_steps": 0.0,
            "price_change_penalty_sum": 0.0,
            "imbalance_penalty_sum": 0.0,
        }
        self.last_arrivals_by_origin = np.zeros(self.n, dtype=float)
        return self._get_obs()

    def _current_slot(self):
        return min(self.period_num - 1, self.t // self.slot_len)

    def _release_transit(self):
        keep_av_service = []
        for arr_t, j, units in self.transit_av_service:
            if arr_t <= self.t:
                self.av_units[j] += units
            else:
                keep_av_service.append((arr_t, j, units))
        self.transit_av_service = keep_av_service

        keep_av_rebalance = []
        for arr_t, j, units in self.transit_av_rebalance:
            if arr_t <= self.t:
                self.av_units[j] += units
            else:
                keep_av_rebalance.append((arr_t, j, units))
        self.transit_av_rebalance = keep_av_rebalance

        keep_cs = []
        for arr_t, j, units in self.transit_cs:
            if arr_t <= self.t:
                self.cs_units[j] += units
            else:
                keep_cs.append((arr_t, j, units))
        self.transit_cs = keep_cs
        self.av_units = np.maximum(self.av_units, 0.0)
        self.cs_units = np.maximum(self.cs_units, 0.0)

    def _av_busy_ratio(self) -> float:
        """
        AV utilization defined as busy_av_units / total_av_units.
        Busy AV = in-transit AV units serving orders.
        Rebalance in-transit AV is excluded.
        """
        busy_av_units = (
            float(np.sum([u for _, _, u in self.transit_av_service]))
            if len(self.transit_av_service) > 0
            else 0.0
        )
        total_av_units = float(np.sum(self.av_units) + busy_av_units)
        return busy_av_units / max(total_av_units, 1e-8)

    def _apply_pricing_action(self, price_action):
        if price_action is None:
            return
        arr = np.asarray(price_action, dtype=int)
        if arr.shape != (self.n,):
            raise ValueError(f"price action shape must be ({self.n},), got {arr.shape}")
        arr = np.clip(arr, 0, len(self.config.cs_share_levels) - 1)
        self.price_idx = arr

    def _apply_rebalance_action(self, rebalance_action):
        if rebalance_action is None:
            return 0.0, 0.0, 0.0
        mat = np.asarray(rebalance_action, dtype=float)
        if mat.shape != (self.n, self.n):
            raise ValueError(f"rebalance action shape must be ({self.n},{self.n}), got {mat.shape}")
        mat = np.maximum(mat, 0.0)
        mat = np.where(self.rebalance_mask, mat, 0.0)

        moved_total = 0.0
        moved_km = 0.0
        empty_cost = 0.0
        for i in range(self.n):
            out_req = float(np.sum(mat[i]))
            if out_req <= 0:
                continue
            avail = float(self.av_units[i])
            max_move = max(0.0, float(self.config.rebalance_max_send_ratio)) * avail
            if max_move <= 1e-12:
                continue
            if out_req > max_move:
                mat[i] *= (max_move / out_req)
                out_req = max_move
            scale = min(1.0, avail / out_req) if out_req > 0 else 0.0
            row = mat[i] * scale
            self.av_units[i] -= np.sum(row)
            for j in range(self.n):
                units = float(row[j])
                if units <= 0:
                    continue
                arr_t = self.t + int(self.travel_steps[i, j])
                self.transit_av_rebalance.append((arr_t, j, units))
                moved_total += units
                moved_km += units * self.travel_km[i, j]
                base_cost = units * self.travel_km[i, j] * self.config.rebalance_cost_per_unit_km
                cost_mult = (
                    self.config.rebalance_adj_cost_mult
                    if bool(self.adj_mask[i, j])
                    else self.config.rebalance_non_adj_cost_mult
                )
                empty_cost += base_cost * cost_mult

        self.metrics["rebalance_sum"] += moved_total
        self.metrics["rebalance_km_sum"] += moved_km
        self.metrics["empty_cost_sum"] += empty_cost
        self.av_units = np.maximum(self.av_units, 0.0)
        return empty_cost, moved_km, moved_total

    def _update_cs_supply_by_price(self):
        # Mean-reverting CS supply response to surge pricing.
        # Higher surge tends to attract more CS supply, with mild elasticity and saturation.
        surge = np.array([self.config.cs_share_levels[idx] for idx in self.price_idx], dtype=float)
        multiplier = np.power(np.maximum(surge, 1e-6), self.config.cs_supply_price_elasticity)
        multiplier = np.clip(
            multiplier,
            self.config.cs_supply_min_multiplier,
            self.config.cs_supply_max_multiplier,
        )
        target = self.cs_base_units * multiplier
        self.cs_units = (1.0 - self.config.cs_recover_rate) * self.cs_units + self.config.cs_recover_rate * target
        self.cs_units = np.maximum(self.cs_units, 0.0)

    def _total_av_units(self) -> float:
        transit_service = float(np.sum([u for _, _, u in self.transit_av_service])) if self.transit_av_service else 0.0
        transit_rebalance = float(np.sum([u for _, _, u in self.transit_av_rebalance])) if self.transit_av_rebalance else 0.0
        return float(np.sum(self.av_units) + transit_service + transit_rebalance)

    def _period_factor(self, slot: int) -> float:
        fac = np.asarray(self.config.demand_period_factors, dtype=float)
        if fac.size == 0:
            return 1.0
        if fac.size == self.period_num:
            return float(max(0.05, fac[slot]))
        x_old = np.linspace(0.0, 1.0, num=fac.size)
        x_new = np.linspace(0.0, 1.0, num=self.period_num)
        mapped = np.interp(x_new, x_old, fac)
        return float(max(0.05, mapped[slot]))

    def _sample_arrivals(self):
        s = self._current_slot()
        demand = self.base_od[s] * self.config.demand_multiplier * self._period_factor(s)
        # Demand-side surge elasticity: higher price multiplier suppresses demand.
        surge = np.array([self.config.cs_share_levels[idx] for idx in self.price_idx], dtype=float)
        demand_multiplier_by_origin = np.power(np.maximum(surge, 1e-6), -self.config.demand_price_elasticity)
        demand = demand * demand_multiplier_by_origin[:, None]
        arrivals = self.rng.poisson(np.maximum(demand, 0.0))
        self.backlog[:, :, 0] += arrivals.astype(float)
        self.last_arrivals_by_origin = np.sum(arrivals, axis=1).astype(float)
        return float(np.sum(arrivals))

    def _serve_orders(self):
        served_total = 0.0
        served_av_total = 0.0
        served_cs_total = 0.0
        platform_revenue = 0.0
        av_cap_total = 0.0
        av_cap_by_origin = np.zeros(self.n, dtype=float)
        served_av_by_origin = np.zeros(self.n, dtype=float)
        served_cs_by_origin = np.zeros(self.n, dtype=float)
        served_total_by_origin = np.zeros(self.n, dtype=float)

        for i in range(self.n):
            av_cap = max(0.0, self.config.av_mu * self.av_units[i])
            cs_cap = max(0.0, self.config.cs_mu * self.cs_units[i])
            av_cap_total += av_cap
            av_cap_by_origin[i] = av_cap
            cap = av_cap + cs_cap
            if cap <= 1e-9:
                continue

            rem_av = av_cap
            rem_cs = cs_cap
            rem_all = cap

            # Oldest-first service for timeout control
            for age in range(self.ttl_steps - 1, -1, -1):
                if rem_all <= 1e-9:
                    break
                for j in range(self.n):
                    q = self.backlog[i, j, age]
                    if q <= 1e-9 or rem_all <= 1e-9:
                        continue
                    x = min(q, rem_all)
                    av_share = rem_av / rem_all if rem_all > 1e-9 else 0.0
                    x_av = min(rem_av, x * av_share)
                    x_cs = min(rem_cs, x - x_av)
                    # numerical safeguard
                    if x_av + x_cs < x:
                        gap = x - x_av - x_cs
                        take = min(gap, rem_av - x_av)
                        x_av += max(0.0, take)
                        x_cs = x - x_av

                    self.backlog[i, j, age] -= x
                    rem_all -= x
                    rem_av -= x_av
                    rem_cs -= x_cs
                    served_total += x
                    served_av_total += x_av
                    served_av_by_origin[i] += x_av
                    served_total_by_origin[i] += x
                    served_cs_total += x_cs
                    served_cs_by_origin[i] += x_cs

                    # move capacity units to destination after travel delay
                    if x_av > 0:
                        out_units = x_av / max(self.config.av_mu, 1e-8)
                        self.av_units[i] -= out_units
                        arr_t = self.t + int(self.travel_steps[i, j])
                        self.transit_av_service.append((arr_t, j, out_units))
                    if x_cs > 0:
                        out_units = x_cs / max(self.config.cs_mu, 1e-8)
                        self.cs_units[i] -= out_units
                        arr_t = self.t + int(self.travel_steps[i, j])
                        self.transit_cs.append((arr_t, j, out_units))

                    dist = max(self.config.min_bill_km, self.travel_km[i, j])
                    fare = self.config.base_fee + self.config.unit_fare_per_km * dist
                    share = self.config.fixed_cs_share
                    if x_av > 0:
                        platform_revenue += x_av * fare
                    if x_cs > 0:
                        platform_revenue += x_cs * fare * (1.0 - share)

        service_cost = (
            served_av_total * self.config.av_order_cost
            + served_cs_total * self.config.cs_order_cost
        )
        util_by_origin = np.divide(
            served_av_by_origin,
            np.maximum(av_cap_by_origin, 1e-8),
        )
        arr_sum = float(np.sum(self.last_arrivals_by_origin))
        if arr_sum > 1e-8:
            w = self.last_arrivals_by_origin / arr_sum
        else:
            cap_sum = float(np.sum(av_cap_by_origin))
            if cap_sum > 1e-8:
                w = av_cap_by_origin / cap_sum
            else:
                w = np.full(self.n, 1.0 / max(self.n, 1), dtype=float)
        demand_weighted_util = float(np.sum(w * util_by_origin))

        self.av_units = np.maximum(self.av_units, 0.0)
        self.cs_units = np.maximum(self.cs_units, 0.0)

        self.metrics["served_sum"] += served_total
        self.metrics["revenue_sum"] += platform_revenue
        self.metrics["demand_util_sum"] += demand_weighted_util
        self.metrics["demand_util_steps"] += 1.0
        return (
            served_total,
            served_av_total,
            served_cs_total,
            platform_revenue,
            service_cost,
            demand_weighted_util,
            served_av_by_origin,
            served_cs_by_origin,
            served_total_by_origin,
        )

    def _timeout_and_age(self):
        timeout_by_origin = np.sum(self.backlog[:, :, self.ttl_steps - 1], axis=1).astype(float)
        timeout_now = float(np.sum(self.backlog[:, :, self.ttl_steps - 1]))
        # remove timed-out then age queue
        self.backlog[:, :, self.ttl_steps - 1] = 0.0
        if self.ttl_steps > 1:
            self.backlog[:, :, 1:] = self.backlog[:, :, :-1]
            self.backlog[:, :, 0] = 0.0
        self.metrics["timeout_sum"] += timeout_now
        return timeout_now, timeout_by_origin

    def _workload_raw(self):
        outstanding = np.sum(self.backlog, axis=(1, 2))
        cap = self.config.av_mu * self.av_units + self.config.cs_mu * self.cs_units
        cap_eff = np.maximum(cap, self.config.workload_cap_floor)
        return outstanding / cap_eff

    def _workload_feature(self):
        w_raw = self._workload_raw()
        w_clip = np.clip(w_raw, 0.0, self.config.workload_clip_max)
        # Log-compressed workload is used by policies to reduce long-tail dominance.
        return np.log1p(w_clip)

    def _workload(self):
        return self._workload_feature()

    def _get_obs(self):
        slot = self._current_slot()
        outstanding = np.sum(self.backlog, axis=(1, 2))
        cap = self.config.av_mu * self.av_units + self.config.cs_mu * self.cs_units
        w_raw = self._workload_raw()
        w = self._workload_feature()
        return {
            "t": int(self.t),
            "slot": int(slot),
            "price_idx": self.price_idx.copy(),
            "av_units": self.av_units.copy(),
            "cs_units": self.cs_units.copy(),
            "outstanding": outstanding.astype(float),
            "capacity": cap.astype(float),
            "workload": w.astype(float),
            "workload_raw": w_raw.astype(float),
        }

    def step(self, action: Dict):
        """
        action:
        {
            "price": np.ndarray shape (n,), optional; only used on pricing boundary
            "rebalance": np.ndarray shape (n,n), optional; only used on rebalance boundary
        }
        """
        if self.t >= self.config.max_steps:
            raise RuntimeError("Episode already done. Call reset().")

        self._release_transit()

        on_pricing_boundary = (self.t % self.config.pricing_interval_steps == 0)
        if on_pricing_boundary:
            prev_price_idx = self.price_idx.copy()
            self._apply_pricing_action(action.get("price", None))
            price_diff = float(np.abs(self.price_idx.astype(float) - prev_price_idx.astype(float)).sum())
            self.last_price_idx = self.price_idx.copy()
        else:
            price_diff = 0.0
        if self.t % self.config.rebalance_interval_steps == 0:
            empty_cost, rebalance_km, rebalance_units = self._apply_rebalance_action(action.get("rebalance", None))
        else:
            empty_cost = 0.0
            rebalance_km = 0.0
            rebalance_units = 0.0

        self._update_cs_supply_by_price()
        arrivals = self._sample_arrivals()
        (
            served,
            served_av,
            served_cs,
            revenue,
            service_cost,
            demand_weighted_util,
            served_av_by_origin,
            served_cs_by_origin,
            served_by_origin,
        ) = self._serve_orders()
        timeout_now, timeout_by_origin = self._timeout_and_age()

        workload = self._workload()
        # Use log scale for robustness so rare pressure spikes don't dominate reward.
        imbalance = float(np.std(np.log1p(workload)))
        av_fixed_cost = self.config.av_fixed_cost_per_unit_step * self._total_av_units()
        # Keep diagnostic metrics, but do not include balancing/utilization terms in reward.
        price_change_penalty = 0.0
        imbalance_penalty = 0.0
        reward = (
            revenue
            - service_cost
            - empty_cost
            - self.config.timeout_penalty * timeout_now
        )

        self.metrics["reward_sum"] += reward
        self.metrics["av_fixed_cost_sum"] += av_fixed_cost
        self.metrics["price_change_penalty_sum"] += 0.0
        self.metrics["imbalance_penalty_sum"] += 0.0
        self.t += 1
        done = self.t >= self.config.max_steps

        info = {
            "arrivals": float(arrivals),
            "served": float(served),
            "served_av": float(served_av),
            "served_cs": float(served_cs),
            "timeout": float(timeout_now),
            "imbalance": float(imbalance),
            "revenue": float(revenue),
            "platform_revenue": float(revenue),
            "demand_weighted_util": float(demand_weighted_util),
            "service_cost": float(service_cost),
            "empty_cost": float(empty_cost),
            "price_change_penalty": 0.0,
            "imbalance_penalty": 0.0,
            "rebalance_km": float(rebalance_km),
            "rebalance_units": float(rebalance_units),
            "av_fixed_cost": float(av_fixed_cost),
            "arrivals_by_origin": self.last_arrivals_by_origin.astype(float),
            "timeout_by_origin": np.asarray(timeout_by_origin, dtype=float),
            "served_av_by_origin": np.asarray(served_av_by_origin, dtype=float),
            "served_cs_by_origin": np.asarray(served_cs_by_origin, dtype=float),
            "served_by_origin": np.asarray(served_by_origin, dtype=float),
            "workload_raw": self._workload_raw().astype(float),
            "reward": float(reward),
            "metrics": self.metrics.copy(),
        }
        return self._get_obs(), float(reward), done, info
