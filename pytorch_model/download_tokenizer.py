from transformers import AutoTokenizer

local_repo_dir = "/home/nyuroprint/.cache/kagglehub/datasets/lizhecheng/llama2-7b-hf/versions/1/Llama2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(local_repo_dir, local_files_only=True)

print("Tokenizer vocab size:", tokenizer.vocab_size)
text = "Hello. What is your name?"
ids = tokenizer(text, return_tensors="pt")["input_ids"]
print(ids)