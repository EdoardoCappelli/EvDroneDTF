from typing import Any


def get_evaluator(config: Any):
    """
    Factory function to create evaluators based on name.
    Args:
        config: Configuration object containing evaluator settings
    Returns:
        Evaluator instance ready to use
    """

    evaluator_registry = {
        'past_conditioned_detr': _create_past_conditioned_detr_evaluator,
    }

    name = config.evaluator_type
    if name not in evaluator_registry:
        available = ', '.join(evaluator_registry.keys())
        raise ValueError(
            f"Unknown evaluator: '{name}'. Available evaluators: {available}"
        )

    evaluator_factory = evaluator_registry[name]
    return evaluator_factory(config)


def _create_past_conditioned_detr_evaluator(config):
    """Create ForecastingEvaluator """
    from evaluation.evaluators.evaluator_rtdetrv2_past_conditioned import PastConditionedEvaluator

    return PastConditionedEvaluator(config)