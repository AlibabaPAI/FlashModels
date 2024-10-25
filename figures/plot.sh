python figures/compare_plot.py --log-files \
log/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp4.log \
log/llama-1b_acc_bs24_seqlen2048_bf16-1_fp16-0_pp1_tp4_fsdp1.log \
dev_llama_auto_sharding.log \
log/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp1.log \
--tags fsdp tp+sp auto dp


# alpaca
python figures/compare_plot.py --log-files \
log-alpaca/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp4.log \
log-alpaca/llama-1b_acc_bs24_seqlen2048_bf16-1_fp16-0_pp1_tp4_fsdp1.log \
dev_llama_auto_sharding_alpaca.log \
log-alpaca/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp1.log \
--tags fsdp tp+sp auto dp \
--title "Alpaca" \
--output-file "alpaca.png"



# wikitext
python figures/compare_plot.py --log-files \
log-wiki/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp4.log \
log-wiki/llama-1b_acc_bs24_seqlen2048_bf16-1_fp16-0_pp1_tp4_fsdp1.log \
dev_llama_auto_sharding.log \
log-wiki/llama-1b_acc_bs6_seqlen2048_bf16-1_fp16-0_pp1_tp1_fsdp1.log \
--tags fsdp tp+sp auto dp \
--title "WikiText" \
--output-file "wikitext.png"