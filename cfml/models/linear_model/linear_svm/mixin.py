import numpy as np

from typing import Any, Dict


class LinearSVMMixin:
    @property
    def w(self):
        return getattr(self, "_w")

    @property
    def lb(self):
        return getattr(self, "_lb")

    def loss_core(self, diff: np.ndarray) -> Dict[str, Any]:
        critical_mask = (diff > 0).ravel()
        loss = 0.5 * np.linalg.norm(self.w)
        has_critical = np.any(critical_mask)
        if has_critical:
            loss += self.lb * diff[critical_mask].mean()
        return {"loss": loss, "has_critical": has_critical, "critical_mask": critical_mask}


__all__ = ["LinearSVMMixin"]
