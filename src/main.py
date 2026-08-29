# src/main.py
import sys
import os

from configs.config_factory import ConfigFactory
from train.trainer_factory import get_trainer
from evaluation.evaluator_factory import get_evaluator

def main():
    config = ConfigFactory.from_cli()
    args = config
    args.print_args()

    mode = getattr(args, 'mode', 'train')

    if mode in ['train', 'both']:
        trainer = get_trainer(args)
        trainer.train()

    if mode in ['eval', 'both']:
        evaluator = get_evaluator(args)
        if getattr(args, 'autoregressive', 0):
            # closed-loop; se eval_forecasting=1 disegna anche le box FUTURE predette
            evaluator.evaluate_autoregressive()
        elif getattr(args, 'eval_forecasting', 0):
            evaluator.evaluate_forecasting()      # ADE/FDE + viz futuro (test_forecast)
        else:
            evaluator.evaluate()                  # detection mAP

if __name__ == '__main__':
    main()
