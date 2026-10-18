"""
TrackTrack-AR: tracker per l'inferenza closed-loop.

Basato su TrackTrack (Shim et al., "Focusing on Tracks for Online Multi-Object Tracking", CVPR 2025).
"""

from collections import deque
from dataclasses import dataclass, fields
import numpy as np

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3
STATE_NAME = {NEW: "new", TRACKED: "tracked", LOST: "lost", REMOVED: "removed"}


@dataclass
class TTConfig:
    use_past: bool = True        # False = senza query-passato
    # candidati
    low_thr: float = 0.10        # sotto: scartato
    det_thr: float = 0.50        # alta / bassa confidenza
    nms_thr: float = 0.45        # NMS delle standard; le alte soppresse diventano candidati "del"
    # TPA
    match_thr: float = 0.80      # soglia iniziale sul costo (in [0, 1])
    reduce_step: float = 0.05    # a ogni giro la soglia si stringe di tanto
    penalty_low: float = 0.20    # TrackTrack: penalty_p
    penalty_del: float = 0.40    # TrackTrack: penalty_q
    iou_gate: float = 0.10       # IoU delle box allargate sotto la soglia → coppia vietata
    buffer: float = 0.30         # margine base della buffered IoU (frazione di w e h per lato)
    buffer_new: float = 1.00     # margine per un track con una sola osservazione
    buffer_speed: float = 1.00   # margine aggiunto per box/frame di velocità
    buffer_lost: float = 0.10    # margine aggiunto per frame perso
    buffer_max: float = 1.50     # tetto del margine
    hmiou: bool = False          # True = costo HMIoU di TrackTrack, False = 1 - IoU
    w_iou: float = 0.50
    w_app: float = 0.0           # >0 solo se il modello restituisce gli embedding
    w_conf: float = 0.10
    w_angle: float = 0.05
    past_bonus: float = 0.20     # slot passato i → track i
    past_cross: float = 0.20     # slot passato i → altri track
    delta_t: int = 3             # orizzonte per la direzione del moto
    kf_vel: float = 1.0 / 160.0  # rumore di processo della velocità del Kalman
    # TAI e ciclo di vita
    init_thr: float = 0.65       # score minimo per far nascere un track
    tai_track_thr: float = 0.30  # buffered IoU max con un track attivo per nascere
    tai_det_thr: float = 0.55    # buffered IoU max con una detection più sicura (TrackTrack: tai_thr)
    min_hits: int = 2            # osservazioni prima di passare a TRACKED
    new_patience: int = 0        # frame persi tollerati da un track NEW
    max_lost: int = 30           # frame persi prima della rimozione
    track_nms_thr: float = 0.45  # NMS fra track aggiornati (0 = off)
    # query-passato
    query_lost: int = 6          # un track perso resta query-passato per questi frame
    query_conf: float = 0.50     # score medio minimo delle osservazioni reali nella finestra
    coher_thr: float = 0.10      # coerenza minima fra box consecutive della finestra
    interp_gaps: bool = True     # al ritorno, le box previste del buco diventano interpolazione
    output_new: bool = False     # emetti anche i track NEW
    out_thr: float = 0.65        # score minimo delle box emesse (0 = tutte)

    @classmethod
    def from_config(cls, config, prefix="ar_tt_"):
        """Legge i campi dal config del progetto (chiavi 'ar_tt_<nome>'), con i default qui sopra."""
        kw = {}
        for f in fields(cls):
            v = getattr(config, prefix + f.name, None)
            if v is not None:
                kw[f.name] = type(f.default)(v) if not isinstance(f.default, bool) else bool(int(v))
        return cls(**kw)


# geometria (box cxcywh normalizzate)
def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def expand(xyxy, bx, by):
    """Box allargate di bx volte la larghezza e by volte l'altezza per lato."""
    w = (xyxy[:, 2] - xyxy[:, 0])[:, None]
    h = (xyxy[:, 3] - xyxy[:, 1])[:, None]
    return xyxy + np.concatenate([-bx * w, -by * h, bx * w, by * h], axis=1)


def buffered(xyxy, buf):
    """C-BIoU: box allargate di buf per lato."""
    return xyxy if buf <= 0 or len(xyxy) == 0 else expand(xyxy, buf, buf)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _pair_iou(a, b):
    """IoU riga per riga fra due liste di box xyxy della stessa lunghezza."""
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def height_iou(a, b):
    """HIoU (Hybrid-SORT, usata da TrackTrack): sovrapposizione verticale / estensione verticale."""
    inter = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    union = np.maximum(a[:, None, 3], b[None, :, 3]) - np.minimum(a[:, None, 1], b[None, :, 1])
    return np.clip(inter, 0.0, None) / np.maximum(union, 1e-12)


def _median0(a):
    """Mediana per colonne, più veloce di np.median su array piccoli."""
    s = np.sort(a, axis=0)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _greedy_keep(iou, thr):
    """NMS greedy su una matrice IoU ordinata per score: maschera delle box tenute."""
    keep = np.ones(len(iou), dtype=bool)
    for i in range(len(iou)):
        if keep[i]:
            keep[i + 1:] &= iou[i, i + 1:] <= thr
    return keep


def nms(xyxy, scores, thr, *more_thr):
    """NMS greedy; con più soglie ritorna gli indici tenuti per ciascuna."""
    order = np.argsort(-scores)
    iou = iou_matrix(xyxy[order], xyxy[order])
    out = [[int(order[i]) for i in np.flatnonzero(_greedy_keep(iou, th))] for th in (thr,) + more_thr]
    return out if more_thr else out[0]


def tpa_match(cost, thr, step):
    """Track-Perspective-Based Association: a ogni giro si accoppiano i minimi reciproci sotto soglia."""
    cost = cost.copy()
    matches = []
    if cost.size == 0:
        return matches
    while True:
        rmin = cost.argmin(axis=1)
        cmin = cost.argmin(axis=0)
        new = [(i, int(j)) for i, j in enumerate(rmin) if cmin[j] == i and cost[i, j] < thr]
        thr -= step
        if not new:
            return matches
        matches += new
        for i, j in new:
            cost[i, :] = 1.0
            cost[:, j] = 1.0


# Kalman a velocità costante, rumore adattivo allo score (NSA)
class KalmanNSA:
    std_pos, std_vel = 1.0 / 20.0, 1.0 / 160.0
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    @classmethod
    def _scale(cls, mean):
        w, h = max(mean[2], 1e-4), max(mean[3], 1e-4)
        return np.array([w, h, w, h])

    @classmethod
    def initiate(cls, z):
        # incertezza iniziale come in TrackTrack: 2*(w,h) sulla posizione, 10*(w,h) sulla velocità
        mean = np.r_[z, np.zeros(4)]
        s = cls._scale(mean)
        return mean, np.diag(np.r_[2.0 * s, 10.0 * s] ** 2)

    @classmethod
    def predict(cls, mean, cov, std_vel=None):
        s = cls._scale(mean)
        q = np.diag(np.r_[cls.std_pos * s, (cls.std_vel if std_vel is None else std_vel) * s] ** 2)
        mean = cls.F @ mean
        return mean, cls.F @ cov @ cls.F.T + q

    @classmethod
    def update(cls, mean, cov, z, score):
        r = np.diag((cls.std_pos * cls._scale(mean)) ** 2) * max(1.0 - float(score), 1e-3)
        S = cls.H @ cov @ cls.H.T + r
        K = np.linalg.solve(S, (cov @ cls.H.T).T).T
        mean = mean + K @ (np.asarray(z, dtype=np.float64) - cls.H @ mean)
        return mean, cov - K @ S @ K.T


class TTrack:
    """Un drone; `boxes` è la finestra di P box (reali e previste) usata come query-passato, indice 0 = la più vecchia."""

    def __init__(self, tid, box, score, frame, P, emb=None, kind="high"):
        box = np.array(box, dtype=np.float64)
        self.track_id = tid
        self.state = NEW
        self.boxes = deque([box], maxlen=P)
        self.real = deque([True], maxlen=P)
        self.scores = deque([float(score)], maxlen=P)
        self.obs = deque([(frame, box)], maxlen=8)          # osservazioni reali recenti
        self.hits, self.lost = 1, 0
        self.score = self.prev_score = float(score)
        self.kf_mean, self.kf_cov = KalmanNSA.initiate(box)
        self.fc, self.fc_frame = None, None                # forecast appreso (T, 4) e frame di origine
        self.pred = box.copy()
        self.emb = None if emb is None else emb / max(np.linalg.norm(emb), 1e-12)
        self.kind = kind                                   # da quale candidato viene l'ultima box
        self.birth = frame
        self._win = None                                   # cache: (score medio reali, coerenza min)
        self._dir = {}                                     # cache: direzione e velocità per delta_t

    # -- moto ----------------------------------------------------------------------------------
    def predict(self, frame, std_vel=None):
        self.kf_mean, self.kf_cov = KalmanNSA.predict(self.kf_mean, self.kf_cov, std_vel)
        # forecast appreso se non ci sono osservazioni più recenti, altrimenti Kalman
        k = None if self.fc is None or self.obs[-1][0] > self.fc_frame else frame - self.fc_frame - 1
        if k is not None and 0 <= k < len(self.fc):
            self.pred = np.asarray(self.fc[k], dtype=np.float64).copy()
        else:
            self.pred = self.kf_mean[:4].copy()
        self.pred[2:] = np.maximum(self.pred[2:], 1e-4)
        self.pred = np.clip(self.pred, 0.0, 1.0)

    def direction(self, delta_t):
        """Direzione del moto (versore dei centri), media pesata su 1..delta_t frame (OC-SORT)."""
        if delta_t in self._dir:
            return self._dir[delta_t]
        v = None
        if len(self.obs) >= 2:
            f_last, b_last = self.obs[-1]
            acc = np.zeros(2)
            for d in range(1, delta_t + 1):
                prev = next((b for f, b in reversed(self.obs) if f <= f_last - d), self.obs[0][1])
                dv = b_last[:2] - prev[:2]
                n = np.linalg.norm(dv)
                if n > 1e-9:
                    acc += dv / n / d
            n = np.linalg.norm(acc)
            v = acc / n if n > 1e-9 else None
        self._dir[delta_t] = v
        return v

    def velocity(self, delta_t):
        """Velocità dei centri per frame fra l'ultima osservazione e quella di delta_t frame prima."""
        key = ("v", delta_t)
        if key not in self._dir:
            v = np.zeros(2)
            if len(self.obs) >= 2:
                f1, b1 = self.obs[-1]
                f0, b0 = next(((f, b) for f, b in reversed(self.obs) if f <= f1 - delta_t), self.obs[0])
                v = (b1[:2] - b0[:2]) / max(f1 - f0, 1)
            self._dir[key] = v
        return self._dir[key]

    def anchor_box(self, delta_t):
        """Osservazione di delta_t frame fa (o la più vecchia disponibile)."""
        f_last = self.obs[-1][0]
        return next((b for f, b in reversed(self.obs) if f <= f_last - delta_t), self.obs[0][1])

    # -- aggiornamenti ---------------------------------------------------------------------------
    def observe(self, box, score, frame, kind, emb, cfg):
        box = np.array(box, dtype=np.float64)
        n = self.lost
        if n > 0 and cfg.interp_gaps:                      # buco ricucito: interpolazione lineare
            last_real = self.obs[-1][1]
            L = len(self.boxes)
            for j in range(1, n + 1):
                idx = L - n - 1 + j
                if 0 <= idx < L and not self.real[idx]:
                    a = j / (n + 1)
                    self.boxes[idx] = (1 - a) * last_real + a * box
        self.boxes.append(box)
        self.real.append(True)
        self.scores.append(float(score))
        self.obs.append((frame, box))
        self.kf_mean, self.kf_cov = KalmanNSA.update(self.kf_mean, self.kf_cov, box, score)
        if emb is not None:
            e = emb / max(np.linalg.norm(emb), 1e-12)
            if self.emb is None:
                self.emb = e
            else:                                          # EMA come TrackTrack: più peso se score alto
                beta = 0.95 + 0.05 * (1.0 - float(score))
                self.emb = beta * self.emb + (1 - beta) * e
                self.emb /= max(np.linalg.norm(self.emb), 1e-12)
        self.prev_score, self.score = self.score, float(score)
        self.hits += 1
        self.lost = 0
        self.kind = kind
        self._win, self._dir = None, {}
        if self.state == LOST or (self.state == NEW and self.hits >= cfg.min_hits):
            self.state = TRACKED

    def miss(self):
        """Frame senza candidato: la finestra prosegue con la box prevista."""
        self.lost += 1
        if self.state == TRACKED:
            self.state = LOST
        self.boxes.append(self.pred.copy())
        self.real.append(False)
        self.scores.append(self.score)
        self.kind = None
        self._win = None

    # -- query-passato -----------------------------------------------------------------------------
    def _window_stats(self, cfg):
        """(score medio delle osservazioni reali, coerenza minima fra box consecutive), in cache."""
        if self._win is None:
            real = [s for s, r in zip(self.scores, self.real) if r]
            mean = float(np.mean(real)) if real else -np.inf
            coh = np.inf
            if len(self.boxes) > 1:
                bx = np.stack(self.boxes)
                # margine come nell'associazione
                spd = _median0(np.abs(np.diff(bx[:, :2], axis=0))) / np.maximum(_median0(bx[:, 2:]), 1e-4)
                mx, my = np.minimum(cfg.buffer + cfg.buffer_speed * spd, max(cfg.buffer, cfg.buffer_max))
                xy = expand(cxcywh_to_xyxy(bx), mx, my)
                step = _pair_iou(xy[:-1], xy[1:])
                if len(bx) > 2:
                    ext = bx[1:-1].copy()
                    ext[:, :2] += bx[1:-1, :2] - bx[:-2, :2]
                    step[1:] = np.maximum(step[1:], _pair_iou(expand(cxcywh_to_xyxy(ext), mx, my), xy[2:]))
                coh = float(step.min())
            self._win = (mean, coh)
        return self._win

    def query_reason(self, P, cfg):
        """None se il track può essere query-passato, altrimenti il motivo."""
        if self.state not in (TRACKED, LOST) or self.lost > cfg.query_lost:
            return "fail_state"
        if len(self.boxes) < P:
            return "fail_len"
        mean, coh = self._window_stats(cfg)
        if mean < cfg.query_conf:
            return "fail_conf"
        if cfg.coher_thr > 0 and coh < cfg.coher_thr:
            return "fail_coher"
        return None

    def trajectory_recent_first(self, P):
        return np.stack(list(self.boxes)[::-1], axis=0)


class TrackTrackAR:
    """Per frame: q = query_tracks(); past = query_array(q); forward; step(...). Al cambio di sequenza: reset()."""

    def __init__(self, cfg: TTConfig, P: int):
        self.cfg, self.P = cfg, int(P)
        self.next_id = 0
        self.stats = {k: 0 for k in ("frames", "m_past", "m_high", "m_low", "m_del", "births",
                                     "tai_track", "tai_det", "recovered", "removed", "track_nms")}
        self.reset()

    def reset(self):
        self.tracks = []
        self.frame = 0

    # -- query-passato ---------------------------------------------------------------------------
    def query_tracks(self):
        if not self.cfg.use_past:
            return []
        return [t for t in self.tracks if t.query_reason(self.P, self.cfg) is None]

    def query_array(self, qtracks):
        if not qtracks:
            return np.zeros((0, self.P, 4), dtype=np.float32)
        return np.stack([t.trajectory_recent_first(self.P) for t in qtracks]).astype(np.float32)

    def query_breakdown(self):
        """Conteggio dei motivi per cui i track NON sono query (diagnostica)."""
        out = {}
        for t in self.tracks:
            r = t.query_reason(self.P, self.cfg)
            if r is not None:
                out[r] = out.get(r, 0) + 1
        return out

    # -- costo --------------------------------------------------------------------------------------
    def _buffer(self, t):
        """Margine C-BIoU per asse: base più velocità osservata; largo per i track nuovi, cresce coi frame persi."""
        cfg = self.cfg
        if t.hits < 2:
            bx = by = max(cfg.buffer, cfg.buffer_new)
        else:
            v = t.velocity(cfg.delta_t)
            bx = cfg.buffer + cfg.buffer_speed * abs(v[0]) / max(t.pred[2], 1e-4)
            by = cfg.buffer + cfg.buffer_speed * abs(v[1]) / max(t.pred[3], 1e-4)
        if t.lost > 0:
            bx += cfg.buffer_lost * t.lost
            by += cfg.buffer_lost * t.lost
        cap = max(cfg.buffer, cfg.buffer_max)
        return min(bx, cap), min(by, cap)

    def _cost(self, tracks, cands):
        cfg = self.cfg
        nt, nc = len(tracks), len(cands)
        if nt == 0 or nc == 0:
            return np.ones((nt, nc))
        pxy = cxcywh_to_xyxy(np.array([t.pred for t in tracks]))
        dxy = cxcywh_to_xyxy(np.array([c["box"] for c in cands]))
        # margine del track applicato alla previsione e ai candidati
        B = np.array([self._buffer(t) for t in tracks])            # (nt, 2): bx, by
        bx, by = B[:, 0:1], B[:, 1:2]
        pw, ph = (pxy[:, 2] - pxy[:, 0])[:, None], (pxy[:, 3] - pxy[:, 1])[:, None]
        pb = pxy + np.concatenate([-bx * pw, -by * ph, bx * pw, by * ph], axis=1)          # (nt, 4)
        dw, dh = (dxy[:, 2] - dxy[:, 0])[None, :], (dxy[:, 3] - dxy[:, 1])[None, :]
        db = dxy[None] + np.stack([-bx * dw, -by * dh, bx * dw, by * dh], axis=2)          # (nt, nc, 4)
        lt = np.maximum(pb[:, None, :2], db[..., :2])
        rb = np.minimum(pb[:, None, 2:], db[..., 2:])
        wh = np.clip(rb - lt, 0.0, None)
        inter = wh[..., 0] * wh[..., 1]
        area_p = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
        area_d = (db[..., 2] - db[..., 0]) * (db[..., 3] - db[..., 1])
        iou = inter / np.maximum(area_p[:, None] + area_d - inter, 1e-12)
        hiou = 1.0
        if cfg.hmiou:                                               # HIoU di TrackTrack (opzione)
            hi = np.minimum(pb[:, None, 3], db[..., 3]) - np.maximum(pb[:, None, 1], db[..., 1])
            hu = np.maximum(pb[:, None, 3], db[..., 3]) - np.minimum(pb[:, None, 1], db[..., 1])
            hiou = np.clip(hi, 0.0, None) / np.maximum(hu, 1e-12)
        d_geo = 1.0 - hiou * iou
        d_app, w_app = 0.0, 0.0
        if cfg.w_app > 0 and all(t.emb is not None for t in tracks) and all(c["emb"] is not None for c in cands):
            te = np.stack([t.emb for t in tracks])
            ce = np.stack([c["emb"] / max(np.linalg.norm(c["emb"]), 1e-12) for c in cands])
            d_app, w_app = np.clip(1.0 - te @ ce.T, 0.0, 1.0), cfg.w_app
        cost = (cfg.w_iou * d_geo + w_app * d_app) / (cfg.w_iou + w_app)
        # confidenza: score previsto contro score del candidato
        s_hat = np.clip(np.array([2 * t.score - t.prev_score for t in tracks]), 0.0, 1.0)
        s_c = np.array([c["score"] for c in cands])
        cost += cfg.w_conf * np.abs(s_hat[:, None] - s_c[None, :])
        # direzione del moto (OC-SORT)
        if cfg.w_angle > 0:
            cc = np.array([c["box"][:2] for c in cands])
            for i, t in enumerate(tracks):
                v = t.direction(cfg.delta_t)
                if v is None:
                    continue
                u = cc - t.anchor_box(cfg.delta_t)[:2]
                n = np.linalg.norm(u, axis=1)
                cosang = np.where(n > 1e-9, (u @ v) / np.maximum(n, 1e-9), 1.0)
                cost[i] += cfg.w_angle * np.arccos(np.clip(cosang, -1, 1)) / np.pi * s_c
        # penalità per tipo di candidato, bonus allo slot passato del proprio track
        kind = np.array([c["kind"] for c in cands])
        past = kind == "past"
        cost[:, (kind == "low") | (past & (s_c <= cfg.det_thr))] += cfg.penalty_low
        cost[:, kind == "del"] += cfg.penalty_del
        if past.any():
            row = {id(t): i for i, t in enumerate(tracks)}
            adj = np.zeros((nt, nc))
            adj[:, past] = cfg.past_cross
            for j in np.flatnonzero(past):
                i = row.get(id(cands[j]["owner"]))
                if i is not None:
                    adj[i, j] = -cfg.past_bonus
            cost += adj
        cost[iou <= cfg.iou_gate] = 1.0
        return np.clip(cost, 0.0, 1.0)

    # -- un frame -----------------------------------------------------------------------------------
    def step(self, past_b, past_s, std_b, std_s, qtracks, future=None, past_emb=None, std_emb=None):
        cfg = self.cfg
        self.frame += 1
        f = self.frame
        self.stats["frames"] += 1
        for t in self.tracks:
            t.predict(f, cfg.kf_vel)

        # candidati: slot passato (legati al proprio track) + standard alte/basse/soppresse
        cands = []
        past_s = np.asarray(past_s, dtype=np.float64).reshape(-1)
        for i, t in enumerate(qtracks):
            if past_s[i] > cfg.low_thr:
                cands.append({"box": np.array(past_b[i], dtype=np.float64), "score": float(past_s[i]),
                              "kind": "past", "owner": t,
                              "emb": None if past_emb is None else np.asarray(past_emb[i], dtype=np.float64)})
        std_s = np.asarray(std_s, dtype=np.float64).reshape(-1)
        std_b = np.asarray(std_b, dtype=np.float64).reshape(-1, 4)
        sel = np.flatnonzero(std_s > cfg.low_thr)
        if sel.size:
            keep, keep_loose = nms(cxcywh_to_xyxy(std_b[sel]), std_s[sel], cfg.nms_thr, 0.95)
            keep_set = set(keep)
            for k in keep:
                kind = "high" if std_s[sel[k]] > cfg.det_thr else "low"
                cands.append(self._cand(std_b, std_s, std_emb, sel[k], kind))
            for k in keep_loose:
                if k not in keep_set and std_s[sel[k]] > cfg.det_thr:
                    cands.append(self._cand(std_b, std_s, std_emb, sel[k], "del"))

        # TPA, stadio 1: track attivi e persi contro tutti i candidati
        act = [t for t in self.tracks if t.state in (TRACKED, LOST)]
        used = set()
        for i, j in tpa_match(self._cost(act, cands), cfg.match_thr, cfg.reduce_step):
            c = cands[j]
            if act[i].state == LOST:
                self.stats["recovered"] += 1
            act[i].observe(c["box"], c["score"], f, c["kind"], c["emb"], cfg)
            self.stats["m_" + c["kind"]] += 1
            used.add(j)
        for t in act:
            if not (t.lost == 0 and t.obs[-1][0] == f):
                t.miss()

        # TPA, stadio 2: track NEW contro le sole standard alte rimaste (come TrackTrack)
        new = [t for t in self.tracks if t.state == NEW]
        if new:
            left = [j for j in range(len(cands)) if j not in used and cands[j]["kind"] == "high"]
            for i, jj in tpa_match(self._cost(new, [cands[j] for j in left]), cfg.match_thr, cfg.reduce_step):
                c = cands[left[jj]]
                new[i].observe(c["box"], c["score"], f, c["kind"], c["emb"], cfg)
                self.stats["m_high"] += 1
                used.add(left[jj])
            for t in new:
                if t.obs[-1][0] != f:
                    t.miss()
                    if t.lost > cfg.new_patience:
                        t.state = REMOVED

        # rimozione dei persi troppo vecchi
        for t in self.tracks:
            if t.state != REMOVED and t.lost > cfg.max_lost:
                t.state = REMOVED
                self.stats["removed"] += 1
        self.tracks = [t for t in self.tracks if t.state != REMOVED]

        # NMS fra track aggiornati in questo frame (buffered IoU): resta il più vecchio
        if cfg.track_nms_thr > 0:
            upd = sorted((t for t in self.tracks if t.lost == 0), key=lambda x: x.track_id)
            if len(upd) > 1:
                xy = buffered(cxcywh_to_xyxy(np.array([t.boxes[-1] for t in upd])), cfg.buffer)
                iou = iou_matrix(xy, xy)
                kept = np.zeros(len(upd), dtype=bool)
                for k, t in enumerate(upd):
                    if kept.any() and iou[k, kept].max() >= cfg.track_nms_thr:
                        t.state = REMOVED
                        self.stats["track_nms"] += 1
                    else:
                        kept[k] = True
                self.tracks = [t for t in self.tracks if t.state != REMOVED]

        # TAI: nascite solo da standard alte e sicure, lontane dai track attivi e da candidati migliori
        init = [j for j in range(len(cands))
                if j not in used and cands[j]["kind"] == "high" and cands[j]["score"] > cfg.init_thr]
        if init:
            init.sort(key=lambda j: -cands[j]["score"])
            ixy = buffered(cxcywh_to_xyxy(np.array([cands[j]["box"] for j in init])), cfg.buffer)
            alive = [t.boxes[-1] for t in self.tracks if t.lost == 0]
            iou_alive = (iou_matrix(ixy, buffered(cxcywh_to_xyxy(np.array(alive)), cfg.buffer)).max(axis=1)
                         if alive else np.zeros(len(init)))
            iou_init = iou_matrix(ixy, ixy)
            born = []
            for k, j in enumerate(init):
                if iou_alive[k] > cfg.tai_track_thr:
                    self.stats["tai_track"] += 1
                    continue
                if born and iou_init[k, born].max() > cfg.tai_det_thr:
                    self.stats["tai_det"] += 1
                    continue
                c = cands[j]
                self.next_id += 1
                t = TTrack(self.next_id, c["box"], c["score"], f, self.P, c["emb"], c["kind"])
                if cfg.min_hits <= 1:
                    t.state = TRACKED
                self.tracks.append(t)
                born.append(k)
                self.stats["births"] += 1

        # forecast degli slot passato: guiderà la previsione dei prossimi frame
        if future is not None:
            for i, t in enumerate(qtracks):
                if t.state != REMOVED:
                    t.fc, t.fc_frame = np.asarray(future[i], dtype=np.float64), f

        dets, src = [], []
        for t in self.tracks:
            if t.lost == 0 and (t.state == TRACKED or (cfg.output_new and t.state == NEW)) and t.score >= cfg.out_thr:
                dets.append((t.boxes[-1], t.score, t.track_id))
                src.append("past" if t.kind == "past" else "standard")
        return dets, src

    @staticmethod
    def _cand(std_b, std_s, std_emb, k, kind):
        return {"box": std_b[k], "score": float(std_s[k]), "kind": kind, "owner": None,
                "emb": None if std_emb is None else np.asarray(std_emb[k], dtype=np.float64)}

    def summary(self):
        s = self.stats
        m = s["m_past"] + s["m_high"] + s["m_low"] + s["m_del"]
        return (f"match {m} (passato {s['m_past']}, alte {s['m_high']}, basse {s['m_low']}, "
                f"soppresse {s['m_del']}) | nascite {s['births']} | TAI: vietate vicino a un track "
                f"{s['tai_track']}, a una detection {s['tai_det']} | track ripresi {s['recovered']} | "
                f"NMS fra track {s['track_nms']}")
