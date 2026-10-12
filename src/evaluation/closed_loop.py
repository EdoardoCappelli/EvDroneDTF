"""
Tracker autoregressivo 'legacy' (closed-loop), condiviso fra:
  - evaluation/evaluators/evaluator_rtdetrv2_past_conditioned.py :: evaluate_autoregressive
  - train/closed_loop_cache.py (rollout del training curriculum: passato prodotto dal modello)
Una sola implementazione → il passato che il modello vede in training è esattamente quello
dell'inferenza (stesse soglie, stessa conferma dei track, stessa NMS, stesso invecchiamento).

Le primitive (Track, best_match_track, nms_std, duplicate_tracks, ...) sono spostate qui senza
modifiche dall'evaluator, che le re-importa (i vecchi import da lì continuano a funzionare).
"""
import numpy as np
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER AUTOREGRESSIVO  (ripreso da create_detection_video.py)
# ══════════════════════════════════════════════════════════════════════════════

def box_iou_cxcywh(a, b):
    """IoU tra due box (4,) cxcywh normalizzate."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def box_center_dist(a, b):
    """Distanza euclidea normalizzata tra i centri di due box cxcywh."""
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


class Track:
    """Finestra scorrevole delle ultime P box predette dal modello (indice 0 = più vecchio)."""

    _next_id = 0   # contatore globale: ogni track nuova riceve un ID univoco e mai riusato

    def __init__(self, box, score, P):
        Track._next_id += 1
        self.track_id = Track._next_id   # identità del tracker (serve per MOTA/IDF1/ID-switch)
        self.boxes = deque([box], maxlen=P)
        self.scores = deque([score], maxlen=P)
        self.missed = 0
        # Proposta C: box t+1 predetta dalla forecasting head al frame precedente
        # (None finché il track non è stato dato come query-passato con head attiva).
        self.forecast_next = None

    def update(self, box, score):
        self.boxes.append(box)
        self.scores.append(score)
        self.missed = 0

    @property
    def last(self):
        return self.boxes[-1]

    @property
    def predicted_next(self):
        """Posizione stimata al frame corrente per l'associazione.

        Proposta C: preferisce la predizione APPRESA dalla forecasting head (t+1 calcolata
        al frame precedente); fallback su estrapolazione lineare se non disponibile.
        """
        if self.forecast_next is not None:
            return np.clip(self.forecast_next, 0.0, 1.0)
        if len(self.boxes) < 2:
            return self.last
        velocity = self.boxes[-1] - self.boxes[-2]
        return np.clip(self.last + velocity, 0.0, 1.0)

    def is_valid(self, P, mean_conf_thr, coher_thr):
        """Valido (inviabile come past) se finestra piena, conf MEDIA alta, box coerenti.
        NB: mean_conf_thr è la soglia sulla MEDIA della finestra — distinta da conf_thr, che
        è la soglia di accettazione della detection CORRENTE (nel loop di update)."""
        if len(self.boxes) < P:
            return False
        if float(np.mean(self.scores)) < mean_conf_thr:
            return False
        bs = list(self.boxes)
        return all(box_iou_cxcywh(a, b) >= coher_thr for a, b in zip(bs[:-1], bs[1:]))

    def trajectory_recent_first(self, P):
        """(P, 4) con indice 0 = più recente — formato atteso dal modello."""
        return np.stack(list(self.boxes)[::-1], axis=0)


def best_match_track(tracks, box, iou_thr, dist_thr, exclude):
    """Track migliore per una detection: IoU su predicted_next (STRETTAMENTE sopra iou_thr),
    poi fallback su distanza centri (sotto dist_thr)."""
    # FIX: era 'v >= best_iou' con best_iou = iou_thr = 0.0. Poiché box_iou_cxcywh >= 0,
    # la condizione era SEMPRE vera → qualunque detection veniva agganciata a un track
    # (all'ultimo della lista, a IoU nulla) e il fallback su distanza restava codice morto.
    # Con '>' una box a IoU 0 (o pari a iou_thr) non aggancia, e il fallback torna vivo.
    best_iou, best_iou_track = iou_thr, None
    for t in tracks:
        if t in exclude:
            continue
        v = box_iou_cxcywh(t.predicted_next, box)
        if v > best_iou:
            best_iou, best_iou_track = v, t
    if best_iou_track is not None:
        return best_iou_track

    best_dist, best_dist_track = dist_thr, None
    for t in tracks:
        if t in exclude:
            continue
        d = box_center_dist(t.predicted_next, box)
        if d < best_dist:
            best_dist, best_dist_track = d, t
    return best_dist_track


def nms_std(std_b, std_s, conf_thr, nms_thr):
    """NMS sulle detection standard sopra conf_thr, PRIMA dell'associazione: un doppione
    (IoU ≥ nms_thr) di una standard più forte sullo stesso drone non deve agganciare né creare
    un track. Ritorna le detection tenute, in ordine di score decrescente."""
    keep = []
    for i in np.argsort(-std_s):
        if std_s[i] <= conf_thr:
            break
        if all(box_iou_cxcywh(std_b[i], std_b[j]) < nms_thr for j in keep):
            keep.append(i)
    return std_b[keep], std_s[keep]


def duplicate_tracks(tracks, updated, nms_thr):
    """NMS fra track: fra quelli aggiornati in questo frame, un track la cui box si sovrappone
    (IoU ≥ nms_thr) a quella di un track PIÙ VECCHIO è un doppione dello stesso drone (es. due
    query-passato convergono sullo stesso drone). `tracks` è in ordine di creazione → vince
    sempre l'identità più vecchia. Ritorna l'insieme dei doppioni da eliminare."""
    kept, dup = [], set()
    for t in tracks:
        if t in updated:
            if any(box_iou_cxcywh(t.last, k.last) >= nms_thr for k in kept):
                dup.add(t)
            else:
                kept.append(t)
    return dup


# ══════════════════════════════════════════════════════════════════════════════
# STATO + PASSO PER-FRAME DEL TRACKER (legacy)
# ══════════════════════════════════════════════════════════════════════════════

class LegacyTracker:
    """Stato del tracker su UNA sequenza (reset() al cambio di sequenza) + il passo per-frame.

    Per ogni frame:
        valid = tracker.query_tracks()                 # track confermati → query-passato
        past  = tracker.query_array(valid)             # (N, P, 4), indice 0 = box più recente
        ... forward 'both' con past ...
        dets, src, stats = tracker.step(past_b, past_s, std_b, std_s, valid, fc_next)

    In oracle (ar_oracle_past=1) il passato viene dalla GT e non dai track: query_tracks/
    query_array non si usano, e step() tratta le slot del passato come detection prioritarie."""

    def __init__(self, P, conf_thr=0.35, mean_conf_thr=None, coher_thr=0.10, iou_thr=0.2,
                 dist_thr=0.08, max_missed=6, std_past_iou_thr=0.3, nms_thr=0.45,
                 std_box_priority=False, use_fc_ref=False, oracle=False):
        self.P = P
        self.conf_thr = conf_thr
        self.mean_conf_thr = conf_thr if mean_conf_thr is None else mean_conf_thr
        self.coher_thr = coher_thr
        self.iou_thr = iou_thr
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.std_past_iou_thr = std_past_iou_thr
        self.nms_thr = nms_thr
        self.std_box_priority = std_box_priority
        self.use_fc_ref = use_fc_ref
        self.oracle = oracle
        self.tracks = []

    @classmethod
    def from_config(cls, config, P):
        """Stessi parametri (e stessi default) di evaluate_autoregressive."""
        conf_thr = getattr(config, 'ar_conf_thr', 0.35)
        _mc = getattr(config, 'ar_mean_conf_thr', -1.0)
        return cls(
            P,
            conf_thr=conf_thr,
            mean_conf_thr=conf_thr if _mc < 0 else _mc,
            coher_thr=getattr(config, 'ar_coher_thr', 0.10),
            iou_thr=getattr(config, 'ar_iou_thr', 0.2),
            dist_thr=getattr(config, 'ar_dist_thr', 0.08),
            max_missed=getattr(config, 'ar_max_missed', 6),
            std_past_iou_thr=getattr(config, 'ar_std_past_iou_thr', 0.3),
            nms_thr=getattr(config, 'nms_threshold', 0.45),
            std_box_priority=bool(getattr(config, 'ar_std_box_priority', 0)),
            use_fc_ref=bool(getattr(config, 'ar_use_forecast_refpoint', 0)),
            oracle=bool(getattr(config, 'ar_oracle_past', 0)),
        )

    def reset(self):
        self.tracks = []

    def query_tracks(self):
        """Track validi (finestra piena, conf media, coerenza) → query-passato, in ordine di creazione."""
        return [t for t in self.tracks if t.is_valid(self.P, self.mean_conf_thr, self.coher_thr)]

    def query_array(self, valid):
        """Finestre dei track validi (N, P, 4), indice 0 = box più recente. Con use_fc_ref la box più
        recente è sostituita dal forecast appreso t+1 (stima del presente dal frame prima)."""
        if not valid:
            return np.zeros((0, self.P, 4), dtype=np.float32)
        trajs = []
        for t in valid:
            traj = t.trajectory_recent_first(self.P)          # (P, 4), indice 0 = più recente
            if self.use_fc_ref and t.forecast_next is not None:
                traj = traj.copy()
                traj[0] = np.clip(t.forecast_next, 0.0, 1.0)
            trajs.append(traj)
        return np.stack(trajs, axis=0)

    def step(self, past_b, past_s, std_b, std_s, valid, fc_next=None):
        """Aggiorna i track con gli output del forward di un frame.

        past_b/past_s: slot [0:N] (le query-passato: i track `valid`, o le query GT in oracle);
        std_b/std_s:   slot [N:] (le query standard); fc_next: future_boxes[0, :N, 0] (N, 4) o None.
        Ritorna (current_dets, current_dets_src, stats): current_dets = [(box cxcywh, score, track_id)],
        current_dets_src = 'past'|'standard' parallelo, stats = {'std_nms', 'dup_tracks'}."""
        P, conf_thr = self.P, self.conf_thr
        iou_thr, dist_thr = self.iou_thr, self.dist_thr
        std_past_iou_thr, nms_thr = self.std_past_iou_thr, self.nms_thr
        tracks = self.tracks
        stats = {'std_nms': 0, 'dup_tracks': 0}

        # NMS sulle standard prima dell'associazione (tutti i rami): un doppione sullo stesso
        # drone non aggancia né crea un secondo track
        n_std = int((std_s > conf_thr).sum())
        std_b, std_s = nms_std(std_b, std_s, conf_thr, nms_thr)
        stats['std_nms'] = n_std - len(std_s)

        updated = set()
        current_dets = []   # (box cxcywh np, score, track_id) — l'ID serve alle metriche di tracking
        current_dets_src = []  # 'past'|'standard' per ogni det, parallelo a current_dets (diag_fp_source; sola misura)
        past_det_boxes = []  # M4: box piazzate dalle query-passato (priorità sulle standard)

        if self.oracle:
            # ORACLE: le past-query sono condizionate sul passato GT. Non c'è mapping
            # 1-a-1 track↔query (il passato non viene dai track) → le tratto come detection
            # ad ALTA priorità e le associo ai track. Gli ID restano assegnati dal tracker.
            for i in np.argsort(-past_s):
                if past_s[i] <= conf_thr:
                    break
                b, s = past_b[i], float(past_s[i])
                m = best_match_track(tracks, b, iou_thr, dist_thr, exclude=updated)
                if m is not None:
                    m.update(b, s); updated.add(m); tid = m.track_id
                else:
                    nt = Track(b, s, P); tracks.append(nt); updated.add(nt); tid = nt.track_id
                current_dets.append((b, s, tid))
                current_dets_src.append('past')
                past_det_boxes.append(b)
        elif self.std_box_priority:
            # MERGE-FIX: la box STANDARD vince (più fresca, letta dall'immagine); il passato
            # RIEMPIE i buchi (droni che lo standard manca) e dà l'IDENTITÀ. → AR >= standard
            # sulla detection, tracking mantenuto dal passato.
            past_out = {t: (b, float(s)) for t, b, s in zip(valid, past_b, past_s)}
            std_for_track = {}
            # 1) STANDARD prima: associa a un track esistente o spawn (una standard per track)
            for i in np.argsort(-std_s):
                if std_s[i] <= conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                m = best_match_track(tracks, b, iou_thr, dist_thr, exclude=updated)
                if m is not None:
                    std_for_track[m] = (b, s); updated.add(m)
                else:
                    # anti-duplicato: se ricalca la box-passato di un track valido (assoc. fallita)
                    # non spawno un doppione → assegno la standard a quel track (vince comunque)
                    hit = next((t for t in valid if t not in updated
                                and past_out.get(t, (None, 0.0))[1] > conf_thr
                                and box_iou_cxcywh(b, past_out[t][0]) >= std_past_iou_thr), None)
                    if hit is not None:
                        std_for_track[hit] = (b, s); updated.add(hit)
                    else:
                        nt = Track(b, s, P); tracks.append(nt); updated.add(nt)
                        current_dets.append((b, s, nt.track_id)); current_dets_src.append('standard')
            # 2) EMISSIONE per i track esistenti: box standard se c'è, altrimenti il passato (buco)
            for t in tracks:
                if t in std_for_track:
                    b, s = std_for_track[t]; t.update(b, s)
                    current_dets.append((b, s, t.track_id)); current_dets_src.append('standard')
                elif past_out.get(t, (None, 0.0))[1] > conf_thr:
                    b, s = past_out[t]; t.update(b, float(s)); updated.add(t)
                    current_dets.append((b, float(s), t.track_id)); current_dets_src.append('past')
                # else: non emesso questo frame → missed lo incrementa il loop di aging sotto
        else:
            # AR: ogni track valido è aggiornato dalla SUA predizione past (1-a-1)
            for t, b, s in zip(valid, past_b, past_s):
                if s > conf_thr:
                    t.update(b, float(s)); updated.add(t)
                    current_dets.append((b, float(s), t.track_id))
                    current_dets_src.append('past')
                    past_det_boxes.append(b)
                # else: frame perso → lo conta UNA volta il loop di invecchiamento sotto
                # (qui c'era anche 'missed += 1': i track validi morivano dopo max_missed/2 frame)

        # detection standard → associa (eredita l'ID) o crea un nuovo track (nuovo ID)
        # (oracle e AR-current; con std_box_priority lo standard è già gestito sopra)
        if not self.std_box_priority:
            for i in np.argsort(-std_s):
                if std_s[i] <= conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                # M4: scarto la standard se ricalca una box già piazzata da una query-passato
                if any(box_iou_cxcywh(b, pbx) >= std_past_iou_thr for pbx in past_det_boxes):
                    continue
                m = best_match_track(tracks, b, iou_thr, dist_thr, exclude=updated)
                if m is not None:
                    m.update(b, s); updated.add(m)
                    current_dets.append((b, s, m.track_id))
                    current_dets_src.append('standard')
                else:
                    nt = Track(b, s, P); tracks.append(nt); updated.add(nt)
                    current_dets.append((b, s, nt.track_id))
                    current_dets_src.append('standard')

        # NMS fra track (tutti i rami): mai due track sullo stesso drone. Resta il più vecchio;
        # il doppione viene eliminato insieme alla sua detection.
        dup = duplicate_tracks(tracks, updated, nms_thr)
        if dup:
            stats['dup_tracks'] = len(dup)
            dup_ids = {t.track_id for t in dup}
            keep_det = [k for k, d in enumerate(current_dets) if d[2] not in dup_ids]
            current_dets = [current_dets[k] for k in keep_det]
            current_dets_src = [current_dets_src[k] for k in keep_det]
            tracks = [t for t in tracks if t not in dup]

        # invecchiamento + eliminazione track morti
        for t in tracks:
            if t not in updated:
                t.missed += 1
        tracks = [t for t in tracks if t.missed < self.max_missed]

        # ── Proposta C: memorizza il forecast t+1 (appreso) per i track validi ──
        # fc_next[i] è la box t+1 del track valid[i] (stesso ordine delle query-passato). Verrà
        # usata come predicted_next all'associazione del frame dopo. I track NON validi in questo
        # frame azzerano forecast_next (evita di riusare una predizione stantìa → predicted_next
        # torna all'estrapolazione lineare). In oracle non c'è mapping track↔query (il passato è
        # GT) → si salta: predicted_next resta l'estrapolazione lineare (forecast_next None).
        if not self.oracle:
            fc_map = {id(t): fb for t, fb in zip(valid, fc_next)} if fc_next is not None else {}
            for t in tracks:
                t.forecast_next = fc_map.get(id(t), None)

        self.tracks = tracks
        return current_dets, current_dets_src, stats
