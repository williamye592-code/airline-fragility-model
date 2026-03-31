from src.simulation.scenario import SequenceScenario


def expected_cost_hold(scenario: SequenceScenario) -> float:
    return scenario.expected_delay_cost_if_hold


def expected_cost_no_hold(scenario: SequenceScenario) -> float:
    return scenario.predicted_spoilage_prob * scenario.expected_disruption_cost_if_no_hold


def choose_action(scenario: SequenceScenario) -> dict:
    hold_cost = expected_cost_hold(scenario)
    no_hold_cost = expected_cost_no_hold(scenario)

    action = "HOLD" if hold_cost < no_hold_cost else "NO_HOLD"

    return {
        "action": action,
        "hold_cost": hold_cost,
        "no_hold_cost": no_hold_cost,
    }