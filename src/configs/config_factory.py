import argparse
from configs.run_configs.rtdetrv2_multiscale_past_conditioned_config import DefaultArgs as RTDetrPastConditionedArgs

class ConfigFactory:
    """Factory class that selects the correct subclass based on --config."""
    
    @staticmethod
    def from_cli():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str, default="RTDetrPastConditioned",
            choices=["RTDetrPastConditioned"],
            help="Choose experiment configuration type."
        )
        
        parser.add_argument("--durations", type=str, default=None, help="Comma-separated list of durations in ms (e.g., '33,165,330')")
        parser.add_argument("--pretrained_rtdetrv2_path", type=str, default=None, help="Path to pretrained RTDETRv2 checkpoint")

        args, unknown = parser.parse_known_args()

        mapping = {
            "RTDetrPastConditioned": RTDetrPastConditionedArgs,
        }

        cls = mapping.get(args.config)
        if cls is None:
            raise ValueError(f"Unknown config type: {args.config}")

        obj = cls()

        if args.durations is not None:
            durations_list = [int(d.strip()) for d in args.durations.split(',')]
            obj.durations = durations_list
            obj.dataset_args['durations'] = durations_list
            print(f"  Durations override: {durations_list}")
        
        if args.pretrained_rtdetrv2_path is not None:
            obj.pretrained_rtdetrv2_path = args.pretrained_rtdetrv2_path
            if hasattr(obj, 'model_args'):
                obj.model_args['pretrained_rtdetrv2_path'] = args.pretrained_rtdetrv2_path
            print(f"  Pretrained DETR: {args.pretrained_rtdetrv2_path}")

        def _str2bool(s):
            if isinstance(s, bool):
                return s
            return str(s).strip().lower() not in ('0', 'false', 'no', 'n', 'off', '')

        override_parser = argparse.ArgumentParser()
        for k, v in obj.__dict__.items():
            if isinstance(v, (dict, list)):
                continue
            if isinstance(v, bool):        
                arg_type = _str2bool
            elif v is None:
                arg_type = str
            else:
                arg_type = type(v)
            override_parser.add_argument(f"--{k}", type=arg_type, default=None)
        overrides = override_parser.parse_args(unknown)
        for k, v in vars(overrides).items():
            if v is None:
                continue
            obj.__dict__[k] = v
            
            for attr, val in list(obj.__dict__.items()):
                if attr.endswith('_args') and isinstance(val, dict) and k in val:
                    val[k] = v

        # Alignement of num_future_annotations and num_future_steps
        nfs = int(getattr(obj, 'num_future_steps', 0) or 0)
        nfa = int(getattr(obj, 'num_future_annotations', 0) or 0)
        if nfs > 0 and nfa < nfs:
            print(f"[config] num_future_annotations ({nfa}) < num_future_steps ({nfs}): "
                  f"allineo num_future_annotations = {nfs} (altrimenti loss_forecast resta 0 in silenzio).")
            obj.__dict__['num_future_annotations'] = nfs
            for attr, val in list(obj.__dict__.items()):
                if attr.endswith('_args') and isinstance(val, dict) and 'num_future_annotations' in val:
                    val['num_future_annotations'] = nfs

        obj.train_full_run_name = f"{obj.run_name}_{obj.dataset_name}"
        return obj