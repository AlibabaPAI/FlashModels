import transformers

MODEL_NAME = "THUDM/chatglm2-6b"
CACHE_DIR = "./hf_models"

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME,
                                                       cache_dir=CACHE_DIR,
                                                       trust_remote_code=True)
