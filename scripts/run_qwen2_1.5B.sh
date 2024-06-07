./main \
--template_filename_axmodel "qwen2-1.5B-8w16a/qwen_l%d.axmodel" \
--axmodel_num 28 \
--tokenizer_type 1 \
--filename_tokenizer_model qwen.tiktoken \
--bos 0 --eos 0 \
--filename_post_axmodel qwen2-1.5B-8w16a/qwen_post.axmodel \
--filename_tokens_embed qwen2-1.5B-8w16a/model.embed_tokens.weight.bfloat16.bin \
--tokens_embed_num 151936 \
--tokens_embed_size 1536 \
--use_mmap_load_embed 1 \
--live_print 1 \
--continue 1 \
--prompt "$1"
