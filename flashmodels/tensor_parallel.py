import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torchacc as ta
from torchacc.dist.tp import Mesh, mark_sharding


class TPContext:

    def __init__(self) -> None:
        self.pp_num = 1
        self.dp_num = 1
        self.tp_num = 1
        self.sp_num = 1
        self.sp = False
        self.initialized = False

    def init_mesh(self, pp_num, tp_num, dp_num, sp):
        self._create_tp_mesh(pp_num, tp_num, dp_num)
        self.sp = sp
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def enable_sp(self):
        return self.sp

    def _create_tp_mesh(self, pp_num, tp_num, dp_num):
        """r create spmd mesh for tp.
        """
        self.pp_num = pp_num
        self.dp_num = dp_num
        self.tp_num = tp_num
        devices_ids = np.arange(ta.dist.world_size())
        self.tp_mesh = Mesh(devices_ids,
                            (self.pp_num, self.dp_num, self.tp_num),
                            ("pp", "dp", "tp"))

    def tp_mark_sharding(self, t, specs, barrier=False):
        if barrier:
            xm.optimization_barrier_([t])
            t = t.view(t.size())
        mark_sharding(t, self.tp_mesh, specs)
        return t

    def get_sharding_spec(self, specs):
        return xs.ShardingSpec(self.tp_mesh, specs)


_tp_context: TPContext = None


def get_tp_context():
    global _tp_context
    if _tp_context is None:
        _tp_context = TPContext()
    return _tp_context


@torch.fx.wrap
def fx_mark_sharding(t, specs, barrier=False):
    if t.is_meta:
        return t
    context = get_tp_context()
    t = context.tp_mark_sharding(t, specs, barrier)
    return t


@torch.fx.wrap
def fx_register_hook(t, specs, barrier=False):
    if t.is_meta:
        return t
    t.register_hook(lambda grad: fx_mark_sharding(grad, specs, barrier))
    return t


class PatchedLinearFor3D(torch.autograd.Function):
    """
    The linear implementation used in 3D parallelism. Some sharding information
    has been added to the backward pass to avoid the issue of SPMD `dot_handler`
    inserting collective permute when dealing with ReplicateOnLastTileDim.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, old_specs=None, new_specs=None):
        # bias is an optional argument
        ctx.old_specs = old_specs
        ctx.new_specs = new_specs
        ctx.save_for_backward(input, weight, bias)
        with torch.no_grad():
            product = torch.einsum("bij,kj->bik", input, weight)
            if bias is None:
                return product
            return product + bias

    @staticmethod
    def backward(ctx, grad_output):
        old_specs = ctx.old_specs
        new_specs = ctx.new_specs
        context = get_tp_context()
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("bik,kj->bij", grad_output,
                                      weight.to(input.dtype))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum("bik,bij->kj", grad_output, input)
            if old_specs:
                context.tp_mark_sharding(grad_weight, old_specs)
            if new_specs:
                grad_weight = context.tp_mark_sharding(
                    grad_weight, new_specs, barrier=True)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.einsum("bik->k", grad_output)

        return grad_input, grad_weight, grad_bias, None, None


class PatchedLinearForSP(torch.autograd.Function):
    """
    The linear implementation used in sequence parallelism. This linear ensures that
    during the transition from SP to TP, the full activation from fwd is not used in bwd.
    TODO: Merge PatchedLinearForSP and PatchedLinearFor3D into one.
    """

    @staticmethod
    def forward(ctx,
                input,
                weight,
                bias=None,
                sequence_parallel=False,
                tp_mesh=None):
        ctx.tp_mesh = tp_mesh
        ctx.sequence_parallel = sequence_parallel
        ctx.save_for_backward(input, weight, bias)
        if sequence_parallel:
            # The view here allows save_for_backward to store sharded input
            input = input.view(input.size())
            # insert all-gather
            xm.optimization_barrier_([input])
            mark_sharding(input, tp_mesh, (0, None, 2))
        with torch.no_grad():
            product = torch.einsum("bij,kj->bik", input, weight)
            if bias is None:
                return product
            else:
                return product + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # Prevent fwd and bwd all-gather from being scheduled together.
            xm.optimization_barrier_([grad_output, weight])
            grad_input = torch.einsum("bik,kj->bij", grad_output,
                                      weight.to(input.dtype))
        if ctx.needs_input_grad[1]:
            if ctx.sequence_parallel:
                import torch_xla

                # The following check ensures that the stored activations
                # are mark_sharding only once, avoiding multiple all-gathers.
                if torch_xla._XLAC._get_xla_sharding_spec(input) == "":
                    # insert all-gather
                    xm.optimization_barrier_([input, grad_output])
                    mark_sharding(input, ctx.tp_mesh, (0, None, 2))
            grad_weight = torch.einsum("bik,bij->kj", grad_output, input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.einsum("bik->k", grad_output)
        return grad_input, grad_weight, grad_bias, None, None
