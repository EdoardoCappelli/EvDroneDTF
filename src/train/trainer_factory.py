from typing import Any


def get_trainer(config: Any):
    """
    Factory function to create trainers based on name.
    Args:
        config: Configuration object containing trainer settings
    Returns:
        Trainer instance ready to use
    """

    # Map trainer names to their implementations
    trainer_registry = {
        'past_conditioned_detr': _create_past_conditioned_detr_trainer,
        }

    name = config.trainer_type
    if name not in trainer_registry:
        available = ', '.join(trainer_registry.keys())
        raise ValueError(
            f"Unknown trainer: '{name}'. Available trainers: {available}"
        )

    # Create and return the trainer
    trainer_factory = trainer_registry[name]
    return trainer_factory(config)


def _create_past_conditioned_detr_trainer(config):
    """Create past-conditioned detector trainer."""
    from train.trainers.train_rtdetrv2_past_conditioned import PastConditionedTrainer

    return PastConditionedTrainer(config)