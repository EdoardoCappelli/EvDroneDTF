from configs.run_configs.base_config import BaseConfig


class DefaultArgs(BaseConfig):
    """
    Configuration for RT-DETR V2 past-conditioned detection baseline.
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
            # --- Augmentation passo 2 (solo train, solo phase 2) ---
            'use_past_dropout': 0, # 1 = droppa randomicamente il passato di alcuni droni (→ ramo standard)
            'past_dropout_p': 0.3, # probabilità di drop per drone
            'use_fake_past': 0, # 1 = passati finti (negative queries → background)
            'fake_past_p': 0.3, # probabilità che un sample riceva finti
            'fake_max_k': 3, # k finti per sample scelto in [1, fake_max_k]
            'fake_collide_thr': 0.15, # soglia IoU anti-collisione coi droni reali
            # --- Mixed query-mode training (un detector per tutte le modalità) ---
            'use_mixed_query_mode': 0, # 1 = randomizza la query-mode per batch in training
            'p_both': 0.6, # prob. modalità 'both'
            'p_past': 0.2, # prob. modalità 'past_only'   (p_std = 1 - p_both - p_past)
            # --- Riproducibilità ---
            'seed': 42,
            # --- Freeze (detector pretrained + solo past_encoder allenabile) ---
            'freeze_detector': 0, # 1 = congela tutto tranne il modulo 'trainable_when_frozen'
            'pretrained_detector_path': '', # checkpoint detector da caricare (vuoto = pesi correnti)
            'trainable_when_frozen': 'past_encoder', # col freeze, modulo allenabile (forecasting → 'forecasting_head')
            # --- Forecasting (passo 3) ---
            'num_future_steps': 0, # T: step futuri (0 = solo detection)
            'forecast_loss_weight': 1.0,
            'forecast_step_weight_max': 3.0, # weighting progressivo: peso dell'ultimo step futuro (1.0 = uniforme). Contrasta il collasso sul presente
            'present_refine_loss_weight': 1.0, # peso L1+GIoU sul presente refined
            'std_loss_weight': 1.0, # peso della loss ufficiale RT-DETR del ramo standard (in both bilancia vs ramo passato; <1 se il passato sotto-allena)
            'use_present_refine': 1, # 1 = present-refinement ON | 0 = OFF
            'forecast_head_type': 'transformer', # 'transformer' | 'mlp' (baseline semplice)
            'vel_avg_k': 3, # step su cui mediare la velocità (ancora CV); 1 = differenza 2-frame
            'use_cv_anchor': 1, # 0 = niente ancora CV → future = present + delta (il modello impara il moto)
            # --- Shared weights (un unico encoder condiviso tra le durate) ---
            'use_shared_weights': 0, # 1 = un solo branch condiviso (invece di uno per durata)
            'shared_cat': 0, # 1 = batcha le durate in UN forward [M*B,...] (richiede use_shared_weights)
            # --- Maschera a BLOCCHI nel decoder (ablation da riguardare) ---
            'block_diag_decoder_attn': 0,   # 1 = ramo standard standalone anche in 'both' (niente interferenza AR, niente mixed necessario)
        })
        self.testing_args.update({
            'test_dataset_name': 'FRED',
            'test_batch_size': 1,
            'num_workers': 4,
            'autoregressive': 0,
            'eval_forecasting': 0, # 1 = calcola ADE/FDE invece della mAP detection
            'ar_use_forecast_refpoint': 0, # forecast t+1 appreso come ref-point della past query
            
            # --- Autoregressive tracker: soglie di associazione closed-loop ---
            'ar_conf_thr': 0.35, # soglia score per confermare/creare un track
            'ar_coher_thr': 0.10, # IoU minima fra box consecutive (validità del track)
            'ar_iou_thr': 0.2, # IoU minima per associare una detection a un track esistente (0 = bug: match sempre)
            'ar_dist_thr': 0.08, # fallback: distanza max fra centri (norm) quando l'IoU non basta
            'ar_std_past_iou_thr': 0.3, # scarta una detection standard se ricalca (IoU≥) una box da query-passato
            'ar_max_missed': 6, # frame senza aggiornamento prima di eliminare un track
            'ar_seq_subsample': 1, # >1 = valuta l'AR su 1 sequenza ogni N 
            'ar_seq_offset': 0, # sfasa la selezione delle sequenze: offset/stride disgiunti → tuning vs report separati
            'ar_export_mot': 0, # 1 = scrive i track in formato MOTChallenge (mot_tracks/{pred,gt}) per eval_tracker.py
            # --- Temporaneo ---
            'diag_best_iou': 0, # 1 = in evaluate() stampa best-IoU per GT su tutte le pred (nessuna soglia score)
            'diag_fp_source': 0, # 1 = in AR decompone gli FP per ramo di provenienza ('past' vs 'standard')
        })
        self.model_args = {
            'model_name': 'rtdetrv2_past_conditioned',
        }
        self.logging_args.update({
            'run_name': 'RTDetr_PastConditioned',
            'wandb_project': 'rtdetrv2_past_conditioned',
        })

        self.merge_args()