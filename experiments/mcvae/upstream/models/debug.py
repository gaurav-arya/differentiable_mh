"""Opt-in diagnostics for comparing Thin estimator paths.

Set ``THIN_DEBUG_PATHS=1`` to emit one-line JSON records.  The diagnostics
are bounded by ``THIN_DEBUG_MAX_BATCHES`` (default three batches) and are
disabled by default, so ordinary training is unchanged.
"""

import json
import os

import torch


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _tensor_summary(value):
    if not torch.is_tensor(value):
        return value
    detached = value.detach()
    # Torch 1.7 does not implement every isnan/isinf overload for Bool.
    numeric = detached.float() if detached.dtype == torch.bool else detached
    finite = torch.isfinite(numeric)
    finite_count = int(finite.sum().item())
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite": finite_count,
        "nan": int(torch.isnan(numeric).sum().item()),
        "inf": int(torch.isinf(numeric).sum().item()),
        "requires_grad": bool(value.requires_grad),
    }
    if finite_count:
        finite_values = numeric[finite]
        summary.update({
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "max_abs": float(finite_values.abs().max().item()),
        })
    return summary


def _gradient_summary(parameters):
    squared_norms = []
    nonfinite_tensors = 0
    nonfinite_values = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        if not bool(finite.all()):
            nonfinite_tensors += 1
            nonfinite_values += int((~finite).sum().item())
        squared_norms.append(gradient.float().square().sum())
    norm = 0.0 if not squared_norms else float(torch.sqrt(torch.stack(squared_norms).sum()).item())
    return {
        "global_l2": norm,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
    }


def _gradient_value_summary(gradients):
    squared_norms = []
    nonfinite_tensors = 0
    nonfinite_values = 0
    for gradient in gradients:
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all()):
            nonfinite_tensors += 1
            nonfinite_values += int((~finite).sum().item())
        squared_norms.append(gradient.float().square().sum())
    norm = 0.0 if not squared_norms else float(torch.sqrt(torch.stack(squared_norms).sum()).item())
    return {
        "global_l2": norm,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
    }


class PathDebugger:
    """Bounded, machine-readable estimator-path diagnostics."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.enabled = _env_bool("THIN_DEBUG_PATHS", False)
        self.start_batch = int(os.environ.get("THIN_DEBUG_START_BATCH", "0"))
        self.stop_batch = int(os.environ.get("THIN_DEBUG_STOP_BATCH", "-1"))
        self.max_batches = int(os.environ.get("THIN_DEBUG_MAX_BATCHES", "3"))
        self.every = int(os.environ.get("THIN_DEBUG_EVERY", "0"))
        self.fail_fast = _env_bool("THIN_FAIL_FAST_NUMERICS", False)
        self.batch_idx = None
        self.active = False

    def begin(self, batch_idx, tensors=None):
        self.batch_idx = int(batch_idx)
        in_window = self.batch_idx >= self.start_batch and (
            self.stop_batch < 0 or self.batch_idx < self.stop_batch
        )
        offset = self.batch_idx - self.start_batch
        if self.every > 0:
            selected = offset % self.every == 0
        else:
            selected = self.max_batches >= 0 and offset < self.max_batches
        self.active = self.enabled and in_window and selected
        if self.active:
            self.record("batch_start", tensors=tensors)

    def record(self, event, tensors=None, values=None):
        if not self.active:
            return
        record = {
            "event": event,
            "estimator": self.estimator,
            "batch": self.batch_idx,
        }
        if values:
            record.update(values)
        if tensors:
            record["tensors"] = {
                name: _tensor_summary(value)
                for name, value in tensors.items()
                if value is not None
            }
        print("THIN_DEBUG " + json.dumps(record, sort_keys=True), flush=True)

    def gradients(self, event, parameters, values=None):
        if not self.active:
            return
        merged = dict(values or {})
        merged["gradients"] = _gradient_summary(parameters)
        self.record(event, values=merged)

    def probe_gradients(self, event, losses, parameters):
        """Probe component gradients without accumulating them into params."""

        if not self.active:
            return
        summaries = {}
        for name, loss in losses.items():
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            summaries[name] = _gradient_value_summary(gradients)
        self.record(event, values={"component_gradients": summaries})

    def end(self):
        if self.active:
            self.record("batch_end")
        self.active = False
