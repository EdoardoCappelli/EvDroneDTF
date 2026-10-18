"""
Tracker autoregressivo 'legacy', usato dall'eval closed-loop e dal rollout del training (closed_loop_cache).
"""
import numpy as np
from collections import deque


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

    _next_id = 0   # ID univoci, mai riusati

    def __init__(self, box, score, P):
        Track._next_id += 1
        self.track_id = Track._next_id
        self.boxes = deque([box], maxlen=P)
        self.scores = deque([score], maxlen=P)
        self.missed = 0
        self.forecast_next = None    # box t+1 prevista dalla forecasting head al frame precedente

    def update(self, box, score):
        self.boxes.append(box)
        self.scores.append(score)
        self.missed = 0

    @property
    def last(self):
        return self.boxes[-1]

    @property
    def predicted_next(self):
        """Posizione attesa al frame corrente: forecast t+1 se c'è, altrimenti estrapolazione lineare."""
        if self.forecast_next is not None:
            return np.clip(self.forecast_next, 0.0, 1.0)
        if len(self.boxes) < 2:
            return self.last
        velocity = self.boxes[-1] - self.boxes[-2]
        return np.clip(self.last + velocity, 0.0, 1.0)

    def is_valid(self, P, mean_conf_thr, coher_thr):
        """Usabile come query-passato: finestra piena, score medio >= mean_conf_thr, box consecutive coerenti."""
        if len(self.boxes) < P:
            return False
        if float(np.mean(self.scores)) < mean_conf_thr:
            return False
        bs = list(self.boxes)
        return all(box_iou_cxcywh(a, b) >= coher_thr for a, b in zip(bs[:-1], bs[1:]))

    def trajectory_recent_first(self, P):
        """(P, 4) con indice 0 = box più recente."""
        return np.stack(list(self.boxes)[::-1], axis=0)


def best_match_track(tracks, box, iou_thr, dist_thr, exclude):
    """Track per una detection: IoU con predicted_next sopra iou_thr, altrimenti il più vicino entro dist_thr."""
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
    """NMS sulle detection standard sopra conf_thr, in ordine di score decrescente."""
    keep = []
    for i in np.argsort(-std_s):
        if std_s[i] <= conf_thr:
            break
        if all(box_iou_cxcywh(std_b[i], std_b[j]) < nms_thr for j in keep):
            keep.append(i)
    return std_b[keep], std_s[keep]


def duplicate_tracks(tracks, updated, nms_thr):
    """Track aggiornati in questo frame che si sovrappongono (IoU >= nms_thr) a un track più vecchio."""
    kept, dup = [], set()
    for t in tracks:
        if t in updated:
            if any(box_iou_cxcywh(t.last, k.last) >= nms_thr for k in kept):
                dup.add(t)
            else:
                kept.append(t)
    return dup


class LegacyTracker:
    """Stato del tracker su una sequenza (reset() al cambio di sequenza).

    Per ogni frame: valid = query_tracks(); past = query_array(valid); forward; step(...).
    In oracle (ar_oracle_past=1) il passato è GT e step() tratta le slot del passato come detection.
    """

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
        """Track validi, in ordine di creazione."""
        return [t for t in self.tracks if t.is_valid(self.P, self.mean_conf_thr, self.coher_thr)]

    def query_array(self, valid):
        """Finestre (N, P, 4) dei track validi, indice 0 = box più recente."""
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
        """Aggiorna i track con gli output di un frame: slot [0:N] del passato, [N:] standard.
        Ritorna (detection [(box, score, track_id)], sorgente 'past'/'standard' di ognuna, stats)."""
        P, conf_thr = self.P, self.conf_thr
        iou_thr, dist_thr = self.iou_thr, self.dist_thr
        std_past_iou_thr, nms_thr = self.std_past_iou_thr, self.nms_thr
        tracks = self.tracks
        stats = {'std_nms': 0, 'dup_tracks': 0}

        # NMS sulle standard prima dell'associazione
        n_std = int((std_s > conf_thr).sum())
        std_b, std_s = nms_std(std_b, std_s, conf_thr, nms_thr)
        stats['std_nms'] = n_std - len(std_s)

        updated = set()
        current_dets = []      # (box cxcywh, score, track_id)
        current_dets_src = []  # 'past' o 'standard'
        past_det_boxes = []    # box delle query-passato, hanno priorità sulle standard

        if self.oracle:
            # passato GT: le query-passato si associano ai track come detection prioritarie
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
            # vince la box standard; il passato riempie i buchi e dà l'identità
            past_out = {t: (b, float(s)) for t, b, s in zip(valid, past_b, past_s)}
            std_for_track = {}
            # prima le standard: si associano a un track o ne creano uno nuovo
            for i in np.argsort(-std_s):
                if std_s[i] <= conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                m = best_match_track(tracks, b, iou_thr, dist_thr, exclude=updated)
                if m is not None:
                    std_for_track[m] = (b, s); updated.add(m)
                else:
                    # se ricalca la box-passato di un track valido va a quel track, senza crearne uno
                    hit = next((t for t in valid if t not in updated
                                and past_out.get(t, (None, 0.0))[1] > conf_thr
                                and box_iou_cxcywh(b, past_out[t][0]) >= std_past_iou_thr), None)
                    if hit is not None:
                        std_for_track[hit] = (b, s); updated.add(hit)
                    else:
                        nt = Track(b, s, P); tracks.append(nt); updated.add(nt)
                        current_dets.append((b, s, nt.track_id)); current_dets_src.append('standard')
            # poi ogni track emette la sua box standard, altrimenti quella del passato
            for t in tracks:
                if t in std_for_track:
                    b, s = std_for_track[t]; t.update(b, s)
                    current_dets.append((b, s, t.track_id)); current_dets_src.append('standard')
                elif past_out.get(t, (None, 0.0))[1] > conf_thr:
                    b, s = past_out[t]; t.update(b, float(s)); updated.add(t)
                    current_dets.append((b, float(s), t.track_id)); current_dets_src.append('past')
        else:
            # ogni track valido si aggiorna con la sua query-passato
            for t, b, s in zip(valid, past_b, past_s):
                if s > conf_thr:
                    t.update(b, float(s)); updated.add(t)
                    current_dets.append((b, float(s), t.track_id))
                    current_dets_src.append('past')
                    past_det_boxes.append(b)

        # detection standard: si associano a un track o ne creano uno nuovo
        if not self.std_box_priority:
            for i in np.argsort(-std_s):
                if std_s[i] <= conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                # scarta le standard che ricalcano una box delle query-passato
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

        # NMS fra track: resta il più vecchio
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

        # il forecast t+1 dei track validi serve all'associazione del frame dopo
        if not self.oracle:
            fc_map = {id(t): fb for t, fb in zip(valid, fc_next)} if fc_next is not None else {}
            for t in tracks:
                t.forecast_next = fc_map.get(id(t), None)

        self.tracks = tracks
        return current_dets, current_dets_src, stats
