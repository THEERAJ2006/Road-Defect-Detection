"""Periodic training checkpoints.

Ultralytics already writes ``weights/last.pt`` every epoch and overwrites it, so a
run that dies mid-epoch keeps only the last completed epoch. That was not enough:
the ablation sweep was lost twice under system memory pressure (the Linux OOM
killer once, zram thrashing to 12 s/iteration another time), and each time it cost
roughly 45 minutes of progress per arm.

This adds an independent, timestamped snapshot every N epochs so the loss is
bounded by N rather than by whenever the machine happens to fall over. Snapshots
are copies of ``last.pt``, which already carries model, optimizer and scheduler
state, so any of them is a valid resume point.
"""

from __future__ import annotations

import pathlib
import shutil

from ultralytics.utils import LOGGER

CKPT_DIRNAME = "checkpoints"


def checkpoint_every(n_epochs: int, keep: int | None = None):
    """Return an ultralytics ``on_train_epoch_end`` callback snapshotting every N epochs.

    Args:
        n_epochs: snapshot interval in epochs.
        keep: retain only the most recent N snapshots per run (None keeps all).
            Pass None for a reproduction — 150 epochs at interval 5 is 30 files of
            ~12 MB per arm, which is cheap and makes the whole run replayable.

    Returns:
        A callback suitable for ``YOLO.add_callback("on_train_epoch_end", fn)``.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")

    def _callback(trainer) -> None:
        epoch = getattr(trainer, "epoch", 0)
        if epoch % n_epochs and epoch != getattr(trainer, "epochs", 0) - 1:
            return  # not on an interval, and not the final epoch
        src = pathlib.Path(trainer.save_dir) / "weights" / "last.pt"
        if not src.exists():
            return
        dst = pathlib.Path(trainer.save_dir) / CKPT_DIRNAME
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / f"epoch{epoch:04d}.pt"
        # copy, never move: last.pt must stay in place for ultralytics' own resume
        shutil.copy2(src, target)
        if keep is not None:
            snaps = sorted(dst.glob("epoch*.pt"))
            for stale in snaps[:-keep]:
                stale.unlink(missing_ok=True)
        # BaseTrainer carries no logger attribute (checked against 8.4.170), so use the
        # package-level one. Reaching for trainer.logger here killed every arm on the
        # first checkpoint boundary.
        LOGGER.info(f"checkpoint: {target.name}")

    return _callback