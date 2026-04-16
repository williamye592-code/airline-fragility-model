import numpy as np

DELAY_COST_PER_MIN = 100
DISRUPTION_COST = 5000

HOLD_OPTIONS = [0, 3, 5, 10]


def compute_expected_cost(p, hold_minutes):
    return p * DISRUPTION_COST + hold_minutes * DELAY_COST_PER_MIN


def find_best_hold(p):
    costs = {}
    for h in HOLD_OPTIONS:
        costs[h] = compute_expected_cost(p, h)

    best_h = min(costs, key=costs.get)
    return best_h, costs