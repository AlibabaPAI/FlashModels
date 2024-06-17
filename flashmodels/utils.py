import io
import json
import os.path as osp

import torch


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_last_step_from_ckpt(ckpt_dir):
    last_step = 0
    if len(ckpt_dir) > 0:
        last_step_path = osp.join(ckpt_dir, "MAX_STEP")
        if osp.exists(last_step_path):
            last_step = torch.load(last_step_path)
    return last_step
