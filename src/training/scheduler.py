from __future__ import annotations

class BetaScheduler:
    def __init__(
        self,
        beta_start:  float = 0.01,
        beta_end:    float = 0.1,
        beta_epochs: int   = 30,
    ) -> None:
        if beta_start <= 0:
            raise ValueError(f"beta_start must be > 0, got {beta_start}")
        if beta_end < beta_start:
            raise ValueError(
                f"beta_end ({beta_end}) must be >= beta_start ({beta_start})"
            )
        if beta_epochs < 1:
            raise ValueError(f"beta_epochs must be >= 1, got {beta_epochs}")

        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.beta_epochs = beta_epochs

    def get(self, epoch: int) -> float:
        if epoch >= self.beta_epochs:
            return self.beta_end

        t = epoch / max(self.beta_epochs - 1, 1)
        return self.beta_start + t * (self.beta_end - self.beta_start)
    
    @classmethod
    def from_config(cls, config: dict) -> "BetaScheduler":
        t = config["training"]
        return cls(
            beta_start=t["beta_start"],
            beta_end=t["beta_end"],
            beta_epochs=t["beta_epochs"],
        )

    def schedule(self, total_epochs: int) -> list[float]:
        return [self.get(e) for e in range(total_epochs)]

    def __repr__(self) -> str:
        return (
            f"BetaScheduler("
            f"beta_start={self.beta_start}, "
            f"beta_end={self.beta_end}, "
            f"beta_epochs={self.beta_epochs})"
        )