from flashmodels.patch.patch import patch_gemma, patch_llama, patch_lora


def patch_amp():
    import torchacc as ta
    ta.patch_amp()


def patch_peft():
    patch_lora()
