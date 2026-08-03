import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- helper to measure VRAM for a model ----------
def measure_model_memory(model_id, device="cuda"):
    print(f"\n===== Testing {model_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    # prepare a dummy batch (batch_size=1, short sequence)
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # forward + backward (using language modelling loss)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    
    # peak memory allocated during the whole forward/backward step
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GiB
    current_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"Peak VRAM allocated: {peak_mem:.2f} GiB")
    print(f"Current VRAM allocated: {current_mem:.2f} GiB")
    
    # total GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free_mem = total_mem - torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"GPU total memory: {total_mem:.2f} GiB, free after run: {free_mem:.2f} GiB")
    
    # clean up to free memory before next test
    del model, tokenizer, inputs, outputs, loss
    torch.cuda.empty_cache()
    
    return peak_mem, free_mem, total_mem

# ---------- main test ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 1: test 0.5B
peak_05, free_05, total_05 = measure_model_memory("Qwen/Qwen2.5-0.5B", device)

# Step 2: decide if we should test 1.5B
# "Significant" leftover – here we use a threshold of 6 GiB free (adjust to your needs)
threshold_gib = 6.0
if free_05 > threshold_gib:
    print(f"\nFree memory ({free_05:.2f} GiB) > threshold ({threshold_gib} GiB). Trying 1.5B model...")
    peak_15, free_15, total_15 = measure_model_memory("Qwen/Qwen2.5-1.5B", device)
else:
    print(f"\nFree memory ({free_05:.2f} GiB) <= threshold ({threshold_gib} GiB). Not testing larger model.")