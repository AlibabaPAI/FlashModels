from flashmodels.accelerators.acc_baichuan_accelerator import \
    ACCBaiChuanAccelerator
from flashmodels.accelerators.acc_gemma_accelerator import ACCGemmaAccelerator
from flashmodels.accelerators.acc_glm_accelerator import ACCGLMAccelerator
from flashmodels.accelerators.acc_gpt_accelerator import ACCGPTAccelerator
from flashmodels.accelerators.acc_llama_accelerator import ACCLLAMAAccelerator
from flashmodels.accelerators.acc_olmo_accelerator import ACCOlmoAccelerator
from flashmodels.accelerators.acc_qwen_accelerator import ACCQwenAccelerator
from flashmodels.accelerators.accelerator import AcceleratorFactory
from flashmodels.accelerators.cuda_baichuan_accelerator import \
    CUDABaiChuanAccelerator
from flashmodels.accelerators.cuda_gemma_accelerator import \
    CUDAGemmaAccelerator
from flashmodels.accelerators.cuda_glm_accelerator import CUDAGLMAccelerator
from flashmodels.accelerators.cuda_llama_accelerator import \
    CUDALLAMAAccelerator
from flashmodels.accelerators.cuda_olmo_accelerator import CUDAOlmoAccelerator
from flashmodels.accelerators.cuda_qwen_accelerator import CUDAQwenAccelerator
from flashmodels.accelerators.megatron_accelerator import MegatronAccelerator


def accelerate(model, loader, args):
    accelerator = AcceleratorFactory.get(args.accelerator, args)
    return accelerator.accelerate(model, loader)
