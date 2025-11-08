import os
import random
import numpy as np
import torch
import logging
from transformers import set_seed


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    logger.info(f"Seeding everything with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed, deterministic=True)


def disable_progresscallback_logging() -> None:
    # monkey patch because of impeccable hf transformers logging source code
    # https://github.com/huggingface/transformers/issues/18093
    from transformers import ProgressCallback

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)

    ProgressCallback.on_log = on_log  # type: ignore
