# Heavily inspired by nanoGPT:

import time
import math
import os, sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.tokenizer import BuildTokenizer
from model import GPTModel, GPTConfig

# Cuda Memory Management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
#print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Training Parameters
eval_only = False
max_iters = 60000
eval_interval = 200
log_interval = 1
eval_iters = 20
eval_only = False
grad_clip = 1.0

# Config parameters
CONTEXT_LENGTH = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
BATCH_SIZE = 12
DROPOUT = 0.2

# AdamW Optimizer Parameters
weight_decay = 1e-1
max_learning_rate = 6e-4
beta1 = 0.9
beta2 = 0.95

# Learning Rate Scheduler P arameters
decay_lr = True
warmup_iters = 200
lr_decay_iters = 60000
min_lr = 6e-5

config = {
    "context_length": CONTEXT_LENGTH,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "batch_size": BATCH_SIZE,
    "dropout": DROPOUT,
    "max_learning_rate": max_learning_rate,
}

# System
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type = 'cuda' if 'cuda' in device.type else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
print("Using device: ", device) 
print("Using dtype: ", dtype)

# Wandb logging
wandb_log = True
wandb_project = 'model-training'
wandb_run_name = 'gpt2-char-transformer' + '_run_' + str(time.time())

# Data Ingestion / Storage
out_dir = 'checkpoints'
path = '/home/ubuntu/char_level_model_transformer/data/enwik8'

t0 = time.time()
print(f"Starting to ingest Train/Valid/Test data...")
tokenized_data = BuildTokenizer(path)
print(f"Finished ingesting; Time: {(time.time() - t0) * 1000 :.2f}ms")

# Extract vocab size
VOCAB_SIZE = 2 ** math.ceil(math.log2(tokenized_data.num_unique_chars))

def batchify(data, batch_size = BATCH_SIZE, block_size = CONTEXT_LENGTH):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:   
        x, y = x.to(device), y.to(device)
    return x, y

# Seed
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Model and Optimizer initialization
model = GPTModel(config = GPTConfig(
    vocab_size = VOCAB_SIZE,
    context_length = CONTEXT_LENGTH,
    n_layers = NUM_LAYERS,
    n_hidden = HIDDEN_SIZE,
    n_head = NUM_HEADS,
    dropout = DROPOUT,
    bias = True
)).to(device)

optimizer = model.configure_optimizers(weight_decay, max_learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for data, split in [(tokenized_data.train, 'train'), (tokenized_data.valid, 'val')]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batchify(data)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
gradient_accumulation_steps = 5 * 8

X, Y = batchify(tokenized_data.train)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
iter_num = 0
best_val_loss = 1e9

print("Starting Training Loop...")
print("Vocab Size:", VOCAB_SIZE)

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else max_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
            
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = batchify(tokenized_data.train)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() *  gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break