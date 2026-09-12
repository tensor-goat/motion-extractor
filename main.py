#!/usr/bin/env python3
"""
motion_extractor_0041_ssla_fixed_timebase.py

SSLA-inspired motion extraction for ordinary video.

This is NOT the SSLA-Det detector and does not use its detector checkpoint.
Instead, it adapts the paper's useful mechanisms to motion extraction:

    frame difference -> sparse pseudo-events
    -> Mixture-of-Spaces (overlapping local states)
    -> position-aware projection (PAP)
    -> gated causal linear recurrent attention
    -> scatter / compute / gather
    -> signed current-frame motion surface

Crucially, recurrent state is NEVER drawn at its old location.  State only
modulates events that exist in the CURRENT frame, which avoids displaying
temporal memory as ghost silhouettes.

Dependencies:
    pip install numpy opencv-python torch

Example:
    python motion_extractor_0041_ssla_fixed_timebase.py --source 0
    python motion_extractor_0041_ssla_fixed_timebase.py --source input.mp4 --device cuda

The implementation is an inference-time, analytically initialized adaptation:
the SSLA machinery is real, but the PAP/attention projections are initialized
deterministically so that it works without training on a custom motion dataset.
"""

import argparse
import math
import sys
import time
import threading
from dataclasses import dataclass
from collections import deque
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


@dataclass
class EventBatch:
    features: torch.Tensor   # [N, D]
    x: torch.Tensor          # [N], long
    y: torch.Tensor          # [N], long
    polarity: torch.Tensor   # [N], float {-1,+1}
    magnitude: torch.Tensor  # [N], float [0,1]
    support: torch.Tensor    # [N], current-frame structural support [0,1]


class SparseMoSLinearAttention:
    """
    Inference-time SSLA-style layer.

    Paper/repo correspondence:
      * overlapping local substates       -> Mixture-of-Spaces
      * event duplicated to P*P states    -> sparse state activation
      * offset-dependent linear maps      -> PAP in / PAP out
      * h <- g*h + v ; o <- q*h           -> gated causal linear attention
      * indexed accumulation / readback   -> scatter-compute-gather

    The paper uses learned weights.  Here they are deterministically initialized
    near identity so the layer can be used immediately for motion extraction.
    """

    def __init__(
        self,
        height: int,
        width: int,
        dim: int = 8,
        patch_size: int = 3,
        memory_decay: float = 0.90,
        device: str = "cpu",
    ):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd positive integer")

        self.h = int(height)
        self.w = int(width)
        self.dim = int(dim)
        self.p = int(patch_size)
        self.area = self.p * self.p
        self.device = torch.device(device)
        self.memory_decay = float(memory_decay)

        # Padded state lattice.  Using H+P-1 by W+P-1 gives every pixel exactly
        # P^2 covering substates, including pixels near image boundaries.
        self.state_h = self.h + self.p - 1
        self.state_w = self.w + self.p - 1
        self.num_states = self.state_h * self.state_w

        self.state = torch.zeros(
            (self.num_states, self.dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_update = torch.full(
            (self.num_states,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        self.pap_in, self.pap_out = self._make_pap()
        self.q_weight, self.v_weight, self.g_weight, self.o_weight = (
            self._make_attention_weights()
        )

    def reset(self) -> None:
        self.state.zero_()
        self.last_update.fill_(-1)

    def _make_pap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic position-aware projections.

        They are actual position-specific D x D linear transforms, initialized
        around identity with small direction-sensitive cross terms.  This keeps
        the extractor useful without a training checkpoint while retaining PAP.
        """
        mats_in = []
        mats_out = []
        c = (self.p - 1) / 2.0
        denom = max(c, 1.0)

        for oy in range(self.p):
            for ox in range(self.p):
                rx = (ox - c) / denom
                ry = (oy - c) / denom
                r2 = rx * rx + ry * ry
                spatial = math.exp(-0.35 * r2)

                m = torch.eye(self.dim, dtype=torch.float32)

                # Position-specific diagonal response.
                m[0, 0] *= 0.90 + 0.10 * spatial  # signed temporal change
                m[1, 1] *= 0.80 + 0.20 * spatial  # magnitude
                m[2, 2] *= 0.95                   # polarity

                # Couple relative patch position to image-gradient channels.
                # features[4], features[5] are gx/gy; feature[0] signed change.
                if self.dim >= 7:
                    m[4, 0] += 0.18 * rx
                    m[5, 0] += 0.18 * ry
                    m[6, 4] += 0.08 * rx
                    m[6, 5] += 0.08 * ry

                mats_in.append(m)

                # Gather projection roughly inverts the directional tilt while
                # retaining spatial weighting.
                mo = torch.eye(self.dim, dtype=torch.float32) * spatial
                if self.dim >= 7:
                    mo[0, 4] += 0.08 * rx
                    mo[0, 5] += 0.08 * ry
                mats_out.append(mo)

        return (
            torch.stack(mats_in, dim=0).to(self.device),
            torch.stack(mats_out, dim=0).to(self.device),
        )

    def _make_attention_weights(self):
        # Near-identity q/v/o makes the untrained block stable and interpretable.
        q = torch.eye(self.dim, dtype=torch.float32, device=self.device)
        v = torch.eye(self.dim, dtype=torch.float32, device=self.device)
        o = torch.eye(self.dim, dtype=torch.float32, device=self.device)

        # Repo form: g = sigmoid(W_g u), h = g*h + v.
        # The constant feature (last channel) sets the base memory gate.
        # Strong motion lowers the gate, replacing stale state more aggressively.
        g = torch.zeros(
            (self.dim, self.dim), dtype=torch.float32, device=self.device
        )
        g[:, -1] = _logit(self.memory_decay)
        if self.dim > 1:
            g[:, 1] = -1.35

        return q, v, g, o

    def _state_ids_for_offset(
        self, x: torch.Tensor, y: torch.Tensor, pos_id: int
    ) -> torch.Tensor:
        oy = pos_id // self.p
        ox = pos_id % self.p
        sy = y + oy
        sx = x + ox
        return sy * self.state_w + sx

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        frame_index: int,
    ) -> torch.Tensor:
        """
        Frame-batched scatter-compute-gather.

        Pseudo-events from one video frame share a timestamp, so events reaching
        the same substate are pooled before the recurrent update.  This avoids an
        arbitrary within-frame ordering while preserving causal frame-to-frame
        state.
        """
        n = int(features.shape[0])
        if n == 0:
            return features

        # -------- SCATTER + PAP_in + sparse recurrent UPDATE --------
        # We accumulate one pooled update and one pooled gate per active state.
        update_sum = torch.zeros_like(self.state)
        gate_sum = torch.zeros_like(self.state)
        counts = torch.zeros(
            (self.num_states,), dtype=torch.float32, device=self.device
        )

        # Cache state ids and PAP-projected event embeddings for the gather pass.
        state_ids_by_pos: List[torch.Tensor] = []
        projected_by_pos: List[torch.Tensor] = []

        ones = torch.ones(n, dtype=torch.float32, device=self.device)

        for pos_id in range(self.area):
            ids = self._state_ids_for_offset(x, y, pos_id)
            u = F.linear(features, self.pap_in[pos_id])
            v = F.linear(u, self.v_weight)
            g = torch.sigmoid(F.linear(u, self.g_weight))

            update_sum.index_add_(0, ids, v)
            gate_sum.index_add_(0, ids, g)
            counts.index_add_(0, ids, ones)

            state_ids_by_pos.append(ids)
            projected_by_pos.append(u)

        active_ids = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
        cnt = counts[active_ids].unsqueeze(1)
        pooled_v = update_sum[active_ids] / cnt
        pooled_g = gate_sum[active_ids] / cnt

        # Lazy time decay: only states touched by current events are accessed.
        # This keeps state sparse in time as well as space.
        last = self.last_update[active_ids]
        age = torch.where(
            last < 0,
            torch.ones_like(last),
            torch.clamp(
                torch.tensor(frame_index, device=self.device, dtype=torch.int32)
                - last,
                min=1,
            ),
        ).float()
        lazy_decay = torch.pow(
            torch.tensor(self.memory_decay, device=self.device), age
        ).unsqueeze(1)

        old = self.state[active_ids] * lazy_decay
        self.state[active_ids] = pooled_g * old + pooled_v
        self.last_update[active_ids] = int(frame_index)

        # -------- COMPUTE + PAP_out + GATHER --------
        gathered = torch.zeros_like(features)

        for pos_id in range(self.area):
            ids = state_ids_by_pos[pos_id]
            u = projected_by_pos[pos_id]

            q = F.linear(u, self.q_weight)
            h = self.state[ids]

            # Gated linear attention readout used by the repo:
            # o = q * h, followed by output projection.
            o = F.linear(q * h, self.o_weight)
            o = F.linear(o, self.pap_out[pos_id])
            gathered += o

        gathered /= float(self.area)

        # Residual + normalization as in the repository's attention layer.
        out = F.layer_norm(gathered + features, (self.dim,))
        return out



class SSLAMotionExtractor0041:
    """
    SSLA motion extractor with a FIXED temporal baseline.

    Unlike 0040, process() receives both the current frame and an externally
    selected reference frame whose timestamp is approximately target_dt earlier.
    Therefore event density is not coupled to CPU/GPU processing speed.

    SSLA hidden state is still only used to modulate CURRENT pseudo-events.
    Hidden state itself is never rasterized, preserving the no-ghost design.
    """

    FEATURE_DIM = 8

    def __init__(
        self,
        threshold: float = 5.0,
        gain: float = 3.0,
        max_events: int = 12000,
        attention_scale: float = 0.5,
        patch_sizes: Sequence[int] = (3, 5),
        memory_decay: float = 0.90,
        edge_floor: float = 5.0,
        structure_floor: float = 0.10,
        stabilize: bool = False,
        dt_normalize: bool = True,
        dt_scale_min: float = 0.50,
        dt_scale_max: float = 2.00,
        device: str = "auto",
    ):
        self.threshold = float(threshold)
        self.gain = float(gain)
        self.max_events = int(max_events)
        self.attention_scale = float(attention_scale)
        self.patch_sizes = tuple(int(p) for p in patch_sizes)
        self.memory_decay = float(memory_decay)
        self.edge_floor = float(edge_floor)
        self.structure_floor = float(structure_floor)
        self.stabilize = bool(stabilize)
        self.dt_normalize = bool(dt_normalize)
        self.dt_scale_min = float(dt_scale_min)
        self.dt_scale_max = float(dt_scale_max)
        self.background_mode = "GRAY"

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.layers: List[SparseMoSLinearAttention] = []
        self.work_h = None
        self.work_w = None
        self.frame_index = 0

        self._lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        )

    def reset(self):
        self.frame_index = 0
        for layer in self.layers:
            layer.reset()

    def _ensure_layers(self, h: int, w: int):
        if self.layers and self.work_h == h and self.work_w == w:
            return

        self.work_h, self.work_w = h, w
        self.layers = [
            SparseMoSLinearAttention(
                h,
                w,
                dim=self.FEATURE_DIM,
                patch_size=p,
                memory_decay=self.memory_decay,
                device=str(self.device),
            )
            for p in self.patch_sizes
        ]

    def _align(self, cur_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
        """Align reference to current to suppress camera shake."""
        p0 = cv2.goodFeaturesToTrack(
            cur_gray,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=16,
            blockSize=3,
        )
        if p0 is None or len(p0) < 8:
            return ref_gray

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            cur_gray,
            ref_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=self._lk_criteria,
        )
        if p1 is None or status is None:
            return ref_gray

        good_cur = p0[status.flatten() == 1]
        good_ref = p1[status.flatten() == 1]
        if len(good_cur) < 8:
            return ref_gray

        M, inliers = cv2.estimateAffinePartial2D(
            good_ref,
            good_cur,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
        )
        if M is None or inliers is None or int(inliers.sum()) < 8:
            return ref_gray

        h, w = cur_gray.shape
        return cv2.warpAffine(
            ref_gray,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _to_work_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

        if self.attention_scale != 1.0:
            h, w = gray.shape
            nw = max(32, int(round(w * self.attention_scale)))
            nh = max(24, int(round(h * self.attention_scale)))
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

        return gray

    def _current_structure(self, gray: np.ndarray):
        f = gray.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        support = np.clip((mag - self.edge_floor) / 40.0, 0.0, 1.0)
        support = self.structure_floor + (
            1.0 - self.structure_floor
        ) * support
        return gx, gy, mag, support

    def _make_events(
        self,
        cur: np.ndarray,
        ref: np.ndarray,
        actual_dt: float,
        target_dt: float,
    ):
        """
        Generate frame-derived pseudo-events with a normalized timebase.

        The capture thread chooses ref ~ target_dt behind cur.
        The small residual lag error is compensated by scaling the temporal
        difference by target_dt / actual_dt (with a conservative clamp).
        """
        diff = cur.astype(np.float32) - ref.astype(np.float32)

        dt_scale = 1.0
        if self.dt_normalize and actual_dt > 1e-6 and target_dt > 1e-6:
            dt_scale = target_dt / actual_dt
            dt_scale = float(
                np.clip(dt_scale, self.dt_scale_min, self.dt_scale_max)
            )
            diff = diff * dt_scale

        absdiff = np.abs(diff)
        mask = absdiff >= self.threshold
        ys, xs = np.nonzero(mask)

        raw_count = int(ys.size)
        if raw_count == 0:
            return None, raw_count, 0, dt_scale

        values = absdiff[ys, xs]

        # Retain the strongest current events only when attention load must be
        # bounded.  raw_count remains visible in the HUD.
        if self.max_events > 0 and raw_count > self.max_events:
            keep = np.argpartition(values, -self.max_events)[-self.max_events:]
            ys = ys[keep]
            xs = xs[keep]
            values = values[keep]

        used_count = int(ys.size)

        gx, gy, grad, support = self._current_structure(cur)

        signed = diff[ys, xs] / 255.0
        magnitude = np.clip(values / 255.0, 0.0, 1.0)
        polarity = np.where(
            signed >= 0.0, 1.0, -1.0
        ).astype(np.float32)
        intensity = cur[ys, xs].astype(np.float32) / 127.5 - 1.0
        gx_ev = np.clip(gx[ys, xs] / 255.0, -1.0, 1.0)
        gy_ev = np.clip(gy[ys, xs] / 255.0, -1.0, 1.0)
        grad_ev = np.clip(grad[ys, xs] / 255.0, 0.0, 1.0)
        ones = np.ones_like(signed, dtype=np.float32)

        features = np.stack(
            [
                signed.astype(np.float32),
                magnitude.astype(np.float32),
                polarity,
                intensity,
                gx_ev.astype(np.float32),
                gy_ev.astype(np.float32),
                grad_ev.astype(np.float32),
                ones,
            ],
            axis=1,
        )

        batch = EventBatch(
            features=torch.from_numpy(features).to(self.device),
            x=torch.from_numpy(xs.astype(np.int64)).to(self.device),
            y=torch.from_numpy(ys.astype(np.int64)).to(self.device),
            polarity=torch.from_numpy(polarity).to(self.device),
            magnitude=torch.from_numpy(
                magnitude.astype(np.float32)
            ).to(self.device),
            support=torch.from_numpy(
                support[ys, xs].astype(np.float32)
            ).to(self.device),
        )
        return batch, raw_count, used_count, dt_scale

    @torch.no_grad()
    def _attention(self, batch: EventBatch) -> torch.Tensor:
        base = batch.features
        z = base

        for layer in self.layers:
            z = layer.forward(
                z, batch.x, batch.y, self.frame_index
            )

        base_n = F.normalize(base, dim=1, eps=1e-6)
        z_n = F.normalize(z, dim=1, eps=1e-6)
        alignment = torch.sum(base_n * z_n, dim=1)

        temporal_energy = torch.tanh(
            0.65 * torch.abs(z[:, 0])
            + 0.35 * torch.abs(z[:, 1])
        )

        confidence = torch.sigmoid(
            1.75 * alignment
            + 0.80 * temporal_energy
            - 0.35
        )

        return 0.25 + 0.75 * confidence

    @torch.no_grad()
    def process(
        self,
        frame_bgr: np.ndarray,
        ref_frame_bgr: np.ndarray,
        actual_dt: float,
        target_dt: float,
    ):
        orig_h, orig_w = frame_bgr.shape[:2]

        cur = self._to_work_gray(frame_bgr)
        ref = self._to_work_gray(ref_frame_bgr)

        h, w = cur.shape
        self._ensure_layers(h, w)

        if ref.shape != cur.shape:
            ref = cv2.resize(
                ref, (w, h), interpolation=cv2.INTER_AREA
            )

        if self.stabilize:
            ref = self._align(cur, ref)

        batch, raw_count, used_count, dt_scale = self._make_events(
            cur, ref, actual_dt, target_dt
        )

        if batch is None:
            work_out = np.full(
                cur.shape,
                128 if self.background_mode == "GRAY" else 0,
                dtype=np.uint8,
            )
        else:
            attention = self._attention(batch)

            # The recurrent SSLA state only gates CURRENT events.
            # It is never independently drawn into the output.
            signed_strength = (
                batch.polarity
                * batch.magnitude
                * attention
                * batch.support
                * self.gain
            )

            if self.background_mode == "GRAY":
                out = torch.full(
                    (h * w,),
                    128.0,
                    dtype=torch.float32,
                    device=self.device,
                )
                vals = 128.0 + 127.0 * signed_strength
            else:
                out = torch.zeros(
                    (h * w,),
                    dtype=torch.float32,
                    device=self.device,
                )
                vals = 255.0 * torch.abs(signed_strength)

            pixel_ids = batch.y * w + batch.x
            out[pixel_ids] = torch.clamp(vals, 0.0, 255.0)

            work_out = (
                out.reshape(h, w)
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            # Spatial-only visualization smoothing.  No temporal persistence.
            work_out = cv2.GaussianBlur(work_out, (3, 3), 0.0)

        self.frame_index += 1

        if work_out.shape != (orig_h, orig_w):
            work_out = cv2.resize(
                work_out,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return work_out, {
            "raw_events": raw_count,
            "used_events": used_count,
            "work_size": (w, h),
            "device": str(self.device),
            "actual_dt_ms": actual_dt * 1000.0,
            "target_dt_ms": target_dt * 1000.0,
            "dt_scale": dt_scale,
        }


class FixedLagCapture:
    """
    Continuously capture frames independently of SSLA processing.

    A processing call receives:
        current = newest available frame
        reference = buffered frame nearest `target_lag_s` before current

    This is the important CPU/CUDA fix: the temporal baseline comes from capture
    timestamps, not from however long the previous inference happened to take.
    """

    def __init__(
        self,
        source,
        target_lag_s: float,
        buffer_frames: int = 24,
        realtime_file: bool = True,
    ):
        self.source = source
        self.target_lag_s = float(target_lag_s)
        self.buffer = deque(maxlen=max(4, int(buffer_frames)))
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.eof = False
        self.error = None
        self.seq = 0

        self.is_camera = isinstance(source, int)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source {source}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.source_fps = fps if fps > 1e-3 else 0.0
        self.realtime_file = bool(realtime_file)

        self.thread = threading.Thread(
            target=self._reader,
            name="FixedLagCapture",
            daemon=True,
        )

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        try:
            file_t0 = time.perf_counter()
            file_index = 0

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    with self.cond:
                        self.eof = True
                        self.cond.notify_all()
                    break

                if self.is_camera:
                    timestamp = time.perf_counter()
                else:
                    # Use the media timeline, not inference timing.
                    pos_ms = float(
                        self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    )
                    if pos_ms > 0.0:
                        timestamp = pos_ms / 1000.0
                    elif self.source_fps > 0.0:
                        timestamp = file_index / self.source_fps
                    else:
                        timestamp = float(file_index)

                with self.cond:
                    self.seq += 1
                    self.buffer.append(
                        (self.seq, timestamp, frame)
                    )
                    self.cond.notify_all()

                if (
                    not self.is_camera
                    and self.realtime_file
                    and self.source_fps > 0.0
                ):
                    file_index += 1
                    target_wall = file_t0 + file_index / self.source_fps
                    delay = target_wall - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                else:
                    file_index += 1

        except Exception as exc:
            with self.cond:
                self.error = exc
                self.eof = True
                self.cond.notify_all()
        finally:
            self.cap.release()

    def get_pair(
        self,
        last_seq: int,
        timeout: float = 1.0,
    ):
        deadline = time.perf_counter() + timeout

        with self.cond:
            while True:
                if self.error is not None:
                    raise self.error

                if self.buffer and self.buffer[-1][0] > last_seq:
                    break

                if self.eof:
                    return None

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.cond.wait(timeout=remaining)

            cur_seq, cur_t, cur_frame = self.buffer[-1]

            # Need enough history to establish a meaningful fixed lag.
            candidates = [
                item for item in self.buffer
                if item[1] < cur_t
            ]
            if not candidates:
                return None

            desired_t = cur_t - self.target_lag_s
            ref_seq, ref_t, ref_frame = min(
                candidates,
                key=lambda item: abs(item[1] - desired_t),
            )

            actual_dt = cur_t - ref_t
            if actual_dt <= 1e-6:
                return None

            # Copy while protected so capture can immediately reuse/replace
            # buffer slots without affecting this inference.
            return (
                cur_seq,
                cur_frame.copy(),
                ref_frame.copy(),
                actual_dt,
            )

    def close(self):
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


def _parse_patch_sizes(text: str) -> Tuple[int, ...]:
    vals = tuple(
        int(v.strip()) for v in text.split(",") if v.strip()
    )
    if not vals:
        raise argparse.ArgumentTypeError(
            "at least one patch size is required"
        )
    if any(v < 1 or v % 2 == 0 for v in vals):
        raise argparse.ArgumentTypeError(
            "patch sizes must be odd positive integers"
        )
    return vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SSLA-inspired sparse attention motion extractor with "
            "fixed-timebase asynchronous capture"
        )
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam ID (0,1,...) or video path",
    )
    parser.add_argument(
        "--event-dt-ms",
        type=float,
        default=33.333,
        help=(
            "Target temporal baseline for pseudo-events in milliseconds. "
            "33.333 ~= 30 Hz; 16.667 ~= 60 Hz."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Pseudo-event threshold in grayscale levels",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=3.0,
        help="Displayed motion gain",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=12000,
        help="Maximum events sent into SSLA; 0 = unlimited",
    )
    parser.add_argument(
        "--capture-buffer",
        type=int,
        default=24,
        help="Number of captured frames retained for fixed-lag lookup",
    )
    parser.add_argument(
        "--attention-scale",
        type=float,
        default=0.5,
        help="SSLA spatial scale; 0.5 is faster than 1.0",
    )
    parser.add_argument(
        "--patches",
        type=_parse_patch_sizes,
        default=(3, 5),
        help="Comma-separated MoS patch sizes, e.g. 3,5",
    )
    parser.add_argument(
        "--memory-decay",
        type=float,
        default=0.90,
        help="Causal SSLA state retention, 0..1",
    )
    parser.add_argument(
        "--edge-floor",
        type=float,
        default=5.0,
        help="Current-frame gradient floor for anti-ghost support",
    )
    parser.add_argument(
        "--structure-floor",
        type=float,
        default=0.10,
        help="Minimum current-frame spatial support",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Affine stabilization before event generation",
    )
    parser.add_argument(
        "--no-dt-normalize",
        action="store_true",
        help="Disable residual target_dt/actual_dt normalization",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, ...",
    )
    args = parser.parse_args()

    if args.event_dt_ms <= 0:
        parser.error("--event-dt-ms must be > 0")
    if args.capture_buffer < 4:
        parser.error("--capture-buffer must be >= 4")
    if not (0.05 <= args.attention_scale <= 1.0):
        parser.error(
            "--attention-scale must be between 0.05 and 1.0"
        )
    if not (0.0 < args.memory_decay < 1.0):
        parser.error(
            "--memory-decay must be between 0 and 1"
        )
    if not (0.0 <= args.structure_floor <= 1.0):
        parser.error(
            "--structure-floor must be between 0 and 1"
        )

    src = (
        int(args.source)
        if str(args.source).isdigit()
        else args.source
    )
    target_dt = args.event_dt_ms / 1000.0

    try:
        capture = FixedLagCapture(
            source=src,
            target_lag_s=target_dt,
            buffer_frames=args.capture_buffer,
            realtime_file=True,
        ).start()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    extractor = SSLAMotionExtractor0041(
        threshold=args.threshold,
        gain=args.gain,
        max_events=args.max_events,
        attention_scale=args.attention_scale,
        patch_sizes=args.patches,
        memory_decay=args.memory_decay,
        edge_floor=args.edge_floor,
        structure_floor=args.structure_floor,
        stabilize=args.stabilize,
        dt_normalize=not args.no_dt_normalize,
        device=args.device,
    )

    title = (
        "SSLA Motion 0041 - Fixed Timebase / Sparse Attention"
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    show_hud = True
    paused = False
    last_seq = 0
    last_output = None
    last_stats = {}
    fps_ema = 0.0

    print("\nControls:")
    print("  [ / ]       gain down / up")
    print("  , / .       pseudo-event threshold down / up")
    print("  b           gray / black background")
    print("  s           stabilization on / off")
    print("  r           reset all SSLA temporal states")
    print("  h           HUD on / off")
    print("  SPACE       pause / resume")
    print("  q / ESC     quit")
    print(
        f"\nFixed pseudo-event baseline: "
        f"{args.event_dt_ms:.3f} ms"
    )
    print(
        "HUD shows raw event count separately from the capped "
        "SSLA input count.\n"
    )

    try:
        while True:
            if not paused:
                pair = capture.get_pair(last_seq, timeout=1.0)

                if pair is None:
                    if capture.eof:
                        break

                    # Capture may still be filling its initial history.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

                seq, frame, ref_frame, actual_dt = pair
                last_seq = seq

                t0 = time.perf_counter()
                output, stats = extractor.process(
                    frame,
                    ref_frame,
                    actual_dt=actual_dt,
                    target_dt=target_dt,
                )

                # CUDA kernels are asynchronous. Synchronize before timing so
                # the FPS HUD reflects actual inference completion.
                if extractor.device.type == "cuda":
                    torch.cuda.synchronize(
                        extractor.device
                    )

                infer_dt = max(
                    time.perf_counter() - t0, 1e-6
                )
                fps = 1.0 / infer_dt
                fps_ema = (
                    fps
                    if fps_ema <= 0
                    else 0.90 * fps_ema + 0.10 * fps
                )

                last_output = output
                last_stats = stats

            else:
                if last_output is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue
                output = last_output
                stats = last_stats

            display = cv2.cvtColor(
                output, cv2.COLOR_GRAY2BGR
            )

            if show_hud:
                ws = stats.get(
                    "work_size", ("?", "?")
                )
                raw = stats.get("raw_events", 0)
                used = stats.get("used_events", 0)
                actual_ms = stats.get(
                    "actual_dt_ms", 0.0
                )
                dt_scale = stats.get("dt_scale", 1.0)

                hud1 = (
                    f"SSLA-0041 | inferFPS={fps_ema:5.1f} | "
                    f"raw={raw} used={used} | "
                    f"dev={stats.get('device', '?')}"
                )
                hud2 = (
                    f"lag={actual_ms:5.1f}ms/"
                    f"{args.event_dt_ms:.1f}ms "
                    f"dtScale={dt_scale:.2f} | "
                    f"gain={extractor.gain:.2f} "
                    f"thr={extractor.threshold:.1f}"
                )
                hud3 = (
                    f"scale={extractor.attention_scale:.2f} "
                    f"work={ws[0]}x{ws[1]} | "
                    f"patches={extractor.patch_sizes} | "
                    f"stab={'ON' if extractor.stabilize else 'OFF'}"
                )

                for i, line in enumerate(
                    (hud1, hud2, hud3)
                ):
                    y = 25 + i * 23
                    cv2.putText(
                        display,
                        line,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        line,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.imshow(title, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
                # On resume, immediately take the newest captured frame.
                if not paused:
                    last_seq = 0
            elif key == ord("]"):
                extractor.gain = min(
                    20.0, extractor.gain + 0.25
                )
            elif key == ord("["):
                extractor.gain = max(
                    0.25, extractor.gain - 0.25
                )
            elif key == ord("."):
                extractor.threshold = min(
                    50.0, extractor.threshold + 0.5
                )
            elif key == ord(","):
                extractor.threshold = max(
                    0.5, extractor.threshold - 0.5
                )
            elif key == ord("b"):
                extractor.background_mode = (
                    "BLACK"
                    if extractor.background_mode == "GRAY"
                    else "GRAY"
                )
            elif key == ord("s"):
                extractor.stabilize = (
                    not extractor.stabilize
                )
                extractor.reset()
            elif key == ord("r"):
                extractor.reset()
            elif key == ord("h"):
                show_hud = not show_hud

    finally:
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
