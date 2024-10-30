import contextlib
import datetime
import os
import os.path as osp
import time

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.experimental.xla_sharding as xs
import torchacc as ta
from torchacc import amp
from torchacc.dist.tp import Mesh

from flashmodels.accelerators.accelerator import AcceleratorFactory
from flashmodels.logger import logger
from flashmodels.utils import get_last_step_from_ckpt


class Trainer(object):
    def __init__(self, model, loader, optimizer, lr_scheduler, tokenizer,
                 args):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.device = model.device
        self.args = args
        self.accelerator = AcceleratorFactory.get(args.accelerator, args)
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        if self.args.pp_num > 1:
            self.gradient_accumulation_steps = 1

        self._prepare_profiling()

    def _prepare_profiling(self):
        if self.args.profile and self.args.local_rank == 0:
            # Refer to https://github.com/pytorch/pytorch/issues/60158 to add
            # additional initialization to avoid hanging during profiling in
            # some cases (such as llama 30B).
            with torch.profiler.profile() as prof:
                pass
            current_time = datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S")
            dir_name = f"./{self.args.profile_dir}/{current_time}"
            os.makedirs(dir_name, exist_ok=True)
            # dump config to the profiling folder
            with open(os.path.join(dir_name, "config.json"), "w") as f:
                f.write(str(self.args))
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name))
            self.profiler.start()
            logger.info("Start profiling ...")

    def _call_profiler(self, step):
        if self.args.local_rank == 0 and self.args.profile:
            if step > self.args.profile_stop_step:
                self.profiler.stop()
                self.args.profile = False
                logger.info("Stop profiling .")
            else:
                self.profiler.step()

    def _context_manager(self):
        return torch.cuda.amp.autocast(
            enabled=not (self.args.fsdp_num > 1 and not self.args.spmd_fsdp),
            dtype=torch.float16 if self.args.fp16 else torch.bfloat16,
            cache_enabled=True)

    def _log(self,
             begin_time,
             step,
             epoch,
             loss=0.0,
             maybe_mark_step=(lambda *args: None)):
        if self.args.log_loss:
            maybe_mark_step()
        else:
            loss = 0.0

        if self.args.local_rank == 0:
            time_each_step = (time.time() -
                              begin_time) / self.args.log_interval
            samples_per_step = float(
              self.args.micro_batch_size * self.args.gradient_accumulation_steps \
                  / time_each_step)
            samples_per_step = samples_per_step * self.args.fsdp_num * self.args.dp_num
            begin_time = time.time()
            train_format_string = "[TRAIN] {{epoch: {}, iteration: {}, batch_size: {}," \
                " loss: {:.8f}, throughput: {:.2f} samples/sec}}"
            logger.info(
                train_format_string.format(
                    epoch, int(step / self.gradient_accumulation_steps),
                    self.args.micro_batch_size, loss, samples_per_step))
        return begin_time

    def train(self):
        if self.args.accelerator == "acc":
            self._acc_train()
        elif self.args.accelerator == "cuda":
            self._cuda_train()
        elif self.args.accelerator == "megatron":
            raise NotImplementedError("Megatron is not supported.")
        else:
            raise RuntimeError(f"unkown accelerator: {self.args.accelerator}")

    def _acc_train(self):
        if self.args.fp16 or self.args.bf16:
            self._acc_train_amp()
        else:
            self._acc_train_fp32()

    def _acc_train_fp32(self):

        last_step = get_last_step_from_ckpt(self.args.ckpt_dir)
        max_step = last_step
        total_loss = torch.tensor(0.0).to(self.device)

        def _step(begin, step, batch):
            step += last_step
            if self.args.pp_num > 1:
                loss = self.model.forward_backward(**batch)
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()
            if (self.args.use_zero2
                    or self.args.use_zero3) and not self.args.pp_num > 1:
                self.model = self.accelerator.shard_grad(self.model)
            nonlocal total_loss
            total_loss += loss.clone().detach()
            if step % self.gradient_accumulation_steps == 0:
                if hasattr(self.model, "clip_grad_norm_"):
                    self.model.clip_grad_norm_(self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if step % (self.args.log_interval *
                       self.gradient_accumulation_steps) == 0:
                begin = self._log(begin, step, epoch, total_loss, ta.mark_step)
            if step > last_step and len(self.args.ckpt_dir) > 0 and step % (
                    self.args.ckpt_freq) == 0:
                self._acc_save(step)
            if step % self.gradient_accumulation_steps == 0:
                total_loss.zero_()
            return begin

        loader = ta.AsyncLoader(self.loader, self.device)
        if (self.args.tp_num > 1 and self.args.dp_num > 1
                and not self.args.pp_num > 1) or (self.args.fsdp_num > 1
                                                  and self.args.spmd_fsdp):
            devices_ids = np.arange(self.args.world_size)
            dp_num = self.args.dp_num if self.args.dp_num > 1 else self.args.fsdp_num
            mesh = Mesh(devices_ids, (dp_num, self.args.tp_num), ("x", "y"))
            loader = pl.MpDeviceLoader(self.loader,
                                       self.device,
                                       input_sharding=xs.ShardingSpec(
                                           mesh, (0, None)))

        for epoch in range(0, self.args.num_train_epochs):
            self.model.train()
            begin = time.time()
            for step, batch in enumerate(loader):
                begin = _step(begin, step + 1, batch)
                max_step += 1
                if max_step == self.args.max_train_steps * self.gradient_accumulation_steps:
                    ta.mark_step()
                    break
                self._call_profiler(step)

        if len(self.args.ckpt_dir) > 0:
            self._acc_save(max_step)

    def _acc_train_amp(self):
        scaler = amp.GradScaler() if self.args.fp16 else None

        last_step = get_last_step_from_ckpt(self.args.ckpt_dir)
        max_step = last_step
        total_loss = torch.tensor(0.0).to(self.device)

        def _step(begin, step, batch):
            found_inf = None
            if self.args.force_use_syncfree_adam:
                found_inf = torch.tensor(0,
                                         dtype=torch.float,
                                         device=self.device)
            if last_step > 0:
                step += last_step + 1
            if self.args.pp_num > 1:

                def output_fn(outputs):
                    loss = outputs["loss"]
                    if scaler is not None:
                        return scaler.scale(loss)
                    return loss

                with self._context_manager():
                    loss = self.model.forward_backward(**batch,
                                                       output_fn=output_fn)
            else:
                with self._context_manager():
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if (self.args.use_zero2
                    or self.args.use_zero3) and not self.args.pp_num > 1:
                self.model = self.accelerator.shard_grad(self.model)
            nonlocal total_loss
            total_loss += loss.clone().detach()
            if step % self.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                if hasattr(self.model, "clip_grad_norm_"):
                    self.model.clip_grad_norm_(self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    if self.args.force_use_syncfree_adam:
                        self.optimizer.step(found_inf=found_inf)
                    else:
                        self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if step % (self.args.log_interval *
                       self.gradient_accumulation_steps) == 0:
                begin = self._log(begin, step, epoch, total_loss, ta.mark_step)
            if step > last_step and len(self.args.ckpt_dir) > 0 and step % (
                    self.args.ckpt_freq) == 0:
                self._acc_save(step)
            if step % self.gradient_accumulation_steps == 0:
                total_loss.zero_()
            return begin

        loader = ta.AsyncLoader(self.loader, self.device)
        if (self.args.tp_num > 1 and self.args.dp_num > 1
                and not self.args.pp_num > 1) or (self.args.fsdp_num > 1
                                                  and self.args.spmd_fsdp):
            devices_ids = np.arange(self.args.world_size)
            dp_num = self.args.dp_num if self.args.dp_num > 1 else int(self.args.fsdp_num / self.args.sp_num)
            mesh = Mesh(devices_ids, (dp_num, self.args.tp_num, self.args.sp_num), ("x", "y", "z"))
            loader = pl.MpDeviceLoader(self.loader,
                                       self.device,
                                       input_sharding=xs.ShardingSpec(
                                           mesh, ("x", None)))

        for epoch in range(0, self.args.num_train_epochs):
            self.model.train()
            begin = time.time()
            for step, batch in enumerate(loader):
                begin = _step(begin, step + 1, batch)
                max_step += 1
                if max_step == self.args.max_train_steps * self.gradient_accumulation_steps:
                    ta.mark_step()
                    break

                self._call_profiler(step)
        if len(self.args.ckpt_dir) > 0:
            self._acc_save(max_step)

    def _acc_save(self, step):
        xm.rendezvous("saving_model")
        ta.mark_step()
        ckpt = {
            "model": self.model.state_dict(),
            "shard_metadata": self.model.get_shard_metadata(),
        }
        ckpt_path = osp.join(
            self.args.ckpt_dir,
            f"rank-{xm.get_ordinal()}-of-{ta.dist.world_size()}-step-{step}.pth"
        )
        ta.save(ckpt, ckpt_path, master_only=False)
        self.tokenizer.save_pretrained(
            self.args.ckpt_dir,
            is_main_process=xm.is_master_ordinal(local=False),
            save_function=ta.save)

        xm.rendezvous("saving_optimizer_states")
        ta.save(
            self.optimizer.state_dict(),
            os.path.join(
                self.args.ckpt_dir, f"optimizer_rank{xm.get_ordinal()}"
                f"-of-{ta.dist.world_size()}-step-{step}"))
        ta.save(
            self.lr_scheduler.state_dict(),
            os.path.join(
                self.args.ckpt_dir, f"scheduler_rank{xm.get_ordinal()}"
                f"-of-{ta.dist.world_size()}-step-{step}"))

        # save rng states
        ta.save({"xla": xm.get_rng_state()},
                os.path.join(
                    self.args.ckpt_dir, f"rng_state_{xm.get_ordinal()}"
                    f"-of-{ta.dist.world_size()}-step-{step}.pth"))

        # save max_step
        ta.save(step,
                osp.join(self.args.ckpt_dir, "MAX_STEP"),
                master_only=True)
        ta.mark_step()

        # TODO(wenting.swt): clean expired checkpoints.

    def _cuda_train(self):
        if self.args.fp16 or self.args.bf16:
            self._cuda_train_amp()
        else:
            self._cuda_train_fp32()

    def _cuda_train_amp(self):
        # TODO(wenting.swt): the distributed acclerate strategy should not be anywhere
        # beyond the Accelerator.
        from torch.cuda.amp import GradScaler
        if self.args.fsdp_num > 1:
            from torch.distributed.fsdp.sharded_grad_scaler import \
                ShardedGradScaler as GradScaler

        scaler = GradScaler() if self.args.fp16 else None
        max_step = 0

        def _step(begin, step, batch):
            with self._context_manager():
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                outputs = self.model(**batch)
                loss = outputs["loss"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % self.args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                if hasattr(self.model, "clip_grad_norm_"):
                    self.model.clip_grad_norm_(self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if step % (self.args.log_interval *
                       self.gradient_accumulation_steps) == 0:
                begin = self._log(begin, step, epoch, loss)
            return begin

        for epoch in range(0, self.args.num_train_epochs):
            self.model.train()
            begin = time.time()
            for step, batch in enumerate(self.loader):
                begin = _step(begin, step + 1, batch)
                max_step += 1
                if max_step == self.args.max_train_steps * self.gradient_accumulation_steps:
                    break
                self._call_profiler(step)

    def _cuda_train_fp32(self):

        max_step = 0

        def _step(begin, step, batch):
            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }
            outputs = self.model(**batch)
            loss = outputs["loss"]
            loss.backward()
            if step % self.args.gradient_accumulation_steps == 0:
                if hasattr(self.model, "clip_grad_norm_"):
                    self.model.clip_grad_norm_(self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if step % (self.args.log_interval *
                       self.gradient_accumulation_steps) == 0:
                begin = self._log(begin, step, epoch, loss)
            return begin

        for epoch in range(0, self.args.num_train_epochs):
            self.model.train()
            begin = time.time()
            for step, batch in enumerate(self.loader):
                begin = _step(begin, step + 1, batch)
                max_step += 1
                if max_step == self.args.max_train_steps * self.gradient_accumulation_steps:
                    break

                self._call_profiler(step)
