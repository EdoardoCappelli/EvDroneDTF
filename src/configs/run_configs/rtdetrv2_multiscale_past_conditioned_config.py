from configs.run_configs.base_config import BaseConfig


class DefaultArgs(BaseConfig):
    """
    Configuration for RT-DETR V2 past-conditioned detection baseline.
    Uses past trajectory queries instead of standard learned queries.
    """
    def __init__(self):
        super().__init__()
        self.dataset_args.update({
            'dataset_name': 'FRED',
            'durations': [3, 33, 165, 330],
            'num_past_annotations': 10,
            'num_future_annotations': 0,
            'num_standard_queries': 50,
        })
        self.training_args.update({
            'task_modality': 'tracking',
            'mode': 'train',
            'trainer_type': 'past_conditioned_detr',
            'query_mode': 'both',
            'phase': 1,
            # --- Augmentation (solo train, phase 2) ---
            'use_past_dropout': 0,     # 1 = toglie il passato ad alcuni droni (-> ramo standard)
            'past_dropout_p': 0.3,     # probabilità di drop per drone
            'use_fake_past': 0,        # 1 = inietta passati finti (negative queries -> background)
            'fake_past_p': 0.3,        # probabilità che un sample riceva finti
            'fake_max_k': 3,           # k finti per sample scelto in [1, fake_max_k]
            'fake_collide_thr': 0.15,  # soglia IoU anti-collisione coi droni reali
            # --- Random walk sulle past-box reali ---
            'use_past_aug': 0,     # 1 = perturba le past-box reali
            'past_aug_p': 0.5,     # prob. che un track venga perturbato
            'past_aug_std': 0.05,  # std del passo del random walk (frazione)
            'past_aug_max': 0.2,   # clamp sull'offset cumulato (frazione)
            # --- Rumore AR(1) sul passato, calibrato sull'errore del tracker su FRED ---
            'use_past_noise': 0,     # 1 = rumore sulle past-box reali (solo train, phase 2)
            'past_noise_p': 0.5,     # prob. che un track venga perturbato
            'past_noise_sx': 0.13,   # std errore del centro x, in frazioni della larghezza
            'past_noise_sy': 0.10,   # std errore del centro y, in frazioni dell'altezza
            'past_noise_sw': 0.22,   # std errore relativo della larghezza
            'past_noise_sh': 0.16,   # std errore relativo dell'altezza
            'past_noise_rho': 0.87,  # autocorrelazione dell'errore fra frame consecutivi
            'past_noise_max': 0.6,   # clamp dell'errore (frazione della box)
            # --- Vanish: passato di droni appena spariti come negativi (solo both) ---
            'use_vanish_past': 0,    # 1 = aggiungi le finestre dei droni che spariscono
            'vanish_past_p': 1.0,    # prob. di usare ciascuna finestra disponibile
            'vanish_max_gap': 6,     # frame di assenza coperti (box estrapolate)
            'vanish_keep_empty': 1,  # 1 = tieni anche i frame senza droni
            'vanish_future_check': 1,  # 1 = scarta i droni che ricompaiono nel futuro
            'vanish_border': 0.0,    # >0 = solo uscite dal bordo (distanza normalizzata)
            # --- Curriculum con passato closed-loop (train/closed_loop_cache.py) ---
            #   ramp:        p += self_past_step a ogni epoca con q >= self_past_q_min
            #   product_iou: p = max(q, self_past_floor) * epoca/epoche
            #   product_map: p = max(mAP50, self_past_floor) * epoca/epoche
            'use_self_past': 0,          # 1 = passato closed-loop in training (solo train, phase 2, both)
            'self_past_schedule': 'ramp',  # 'ramp' | 'product_iou' | 'product_map'
            'self_past_floor': 0.0,      # product_*: minimo del fattore di qualità
            'self_past_warmup': 4,       # epoche di solo passato GT prima del primo rollout
            'self_past_refresh': 1,      # rollout ogni N epoche
            'self_past_aggregate': 1,    # 1 = la cache si accumula fra i rollout
            'self_past_subset': 0.3,     # frazione delle sequenze di training per rollout
            'self_past_val_subset': 0.5, # frazione delle sequenze di val su cui misurare q
            'self_past_q_min': 0.6,      # soglia di qualità per far salire p
            'self_past_step': 0.05,      # incremento di p per epoca
            'self_past_p_max': 0.8,      # tetto di p
            'self_past_match_iou': 0.1,  # IoU minima per assegnare il target (sotto -> sfondo)
            'self_past_workers': 4,      # worker del DataLoader del rollout
            'self_past_batch': 16,       # sequenze avanzate insieme nel rollout
            'self_past_max_frames': 0,   # debug: massimo frame per sequenza (0 = tutti)
            # --- Mixed query-mode training ---
            'use_mixed_query_mode': 0,   # 1 = randomizza la query-mode per batch in training
            'p_both': 0.6,               # prob. modalità 'both'
            'p_past': 0.2,               # prob. modalità 'past_only'   (p_std = 1 - p_both - p_past)
            # --- Riproducibilità ---
            'seed': 42,
            # --- Freeze (detector pretrained + solo past_encoder allenabile) ---
            'freeze_detector': 0,            # 1 = congela tutto tranne il modulo 'trainable_when_frozen'
            'pretrained_detector_path': '',  # checkpoint detector da caricare (vuoto = pesi correnti)
            'trainable_when_frozen': 'past_encoder',  # col freeze, modulo allenabile (forecasting -> 'forecasting_head')
            # --- Forecasting ---
            'num_future_steps': 0,           # T: step futuri (0 = solo detection)
            'forecast_loss_weight': 1.0,
            'forecast_step_weight_max': 3.0,      # peso dell'ultimo step futuro (1.0 = uniforme)
            'present_refine_loss_weight': 1.0,    # peso L1+GIoU sul presente raffinato
            'std_loss_weight': 1.0,               # peso della loss RT-DETR del ramo standard
            'use_present_refine': 1,              # 1 = present-refinement, 0 = presente grezzo
            'forecast_head_type': 'transformer',  # 'transformer' | 'mlp' (baseline semplice)
            'vel_avg_k': 3,                       # step su cui mediare la velocità (ancora CV); 1 = differenza 2-frame
            'use_cv_anchor': 1,                   # 0 = future = present + delta, senza ancora CV
            # --- Shared weights (un unico encoder condiviso tra le durate) ---
            'use_shared_weights': 0,   # 1 = un solo branch condiviso (invece di uno per durata)
            'shared_cat': 0,           # 1 = un solo forward per tutte le durate (richiede use_shared_weights)
            # --- Ablation decoder ---
            'block_diag_decoder_attn': 0,   # 1 = maschera a blocchi, niente attenzione past <-> std
            'use_past_class_head': 0,       # 1 = class_embed separata per il ramo passato (uguale in train/eval)
            'select_exclude_cls': 0,        # 1 = best/early-stop su val_loss senza la cls
        })
        self.testing_args.update({
            'test_dataset_name': 'FRED',
            'test_batch_size': 1,
            'num_workers': 4,
            'autoregressive': 0,
            'eval_forecasting': 0,   # 1 = calcola ADE/FDE invece della mAP detection
            'ar_use_forecast_refpoint': 0,  # 1 = forecast t+1 come ref-point della past query
            # --- Tracker autoregressivo ---
            'ar_conf_thr': 0.35,      # soglia score della detection corrente
            'ar_mean_conf_thr': -1.0, # soglia sulla conf media della finestra; -1 = ar_conf_thr
            'ar_std_box_priority': 0, # 1 = vince la box standard, 0 = vince il passato
            'ar_coher_thr': 0.10,     # IoU minima fra box consecutive (validità del track)
            'ar_iou_thr': 0.2,        # IoU minima per associare una detection a un track esistente
            'ar_dist_thr': 0.08,      # fallback: distanza max fra centri (norm) quando l'IoU non basta
            'ar_std_past_iou_thr': 0.3,  # scarta una detection standard sovrapposta a una box del passato
            'ar_max_missed': 6,       # frame senza aggiornamento prima di eliminare un track
            'ar_seq_subsample': 1,    # >1 = AR su 1 sequenza ogni N
            'ar_seq_offset': 0,       # offset della selezione delle sequenze
            'ar_export_mot': 0,       # 1 = salva i track in formato MOTChallenge (mot_tracks/{pred,gt})
            'ar_oracle_past': 0,      # 1 = passato = GT (limite superiore)
            'ar_tracker': 'legacy',   # 'legacy' | 'tracktrack' (evaluation/tracktrack_ar.py)
            # --- Diagnostica ---
            'diag_best_iou': 0,       # 1 = in evaluate() stampa best-IoU per GT su tutte le pred
            'diag_fp_source': 0,      # 1 = in AR divide gli FP per ramo ('past' vs 'standard')
        })
        # parametri di TrackTrack-AR: chiavi 'ar_tt_<nome>', default in TTConfig
        from evaluation.tracktrack_ar import TTConfig
        from dataclasses import fields as _fields
        self.testing_args.update({f'ar_tt_{f.name}': f.default for f in _fields(TTConfig)})
        self.model_args = {
            'model_name': 'rtdetrv2_past_conditioned',
        }
        self.logging_args.update({
            'run_name': 'RTDetr_PastConditioned',
            'wandb_project': 'rtdetrv2_past_conditioned',
        })

        self.merge_args()