from dataclasses import dataclass


@dataclass
class SequenceScenario:
    predicted_spoilage_prob: float
    expected_delay_cost_if_hold: float
    expected_disruption_cost_if_no_hold: float