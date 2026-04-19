import os
from pathlib import Path
from typing import Union

import torch
import tiktoken

from cs336_basics.config import ModelConfig
from cs336_basics.nn_modules import TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _initialize_weights(config: ModelConfig) -> dict[str, torch.Tensor]:
    # Use a standard deviation for initialization (e.g., 0.02 is common for Transformers)
    std = 0.02

    weights: dict[str, torch.Tensor] = {}
    weights['token_embeddings.weight'] = torch.randn(config.vocab_size, config.d_model, device=device) * std
    weights['ln_final.weight'] = torch.ones(config.d_model, device=device)
    weights['lm_head.weight'] = torch.randn(config.vocab_size, config.d_model, device=device) * std

    for index in range(config.num_layers):
        weights[f"layers.{index}.attn.q_proj.weight"] = torch.randn(config.d_model, config.d_model, device=device) * std
        weights[f"layers.{index}.attn.k_proj.weight"] = torch.randn(config.d_model, config.d_model, device=device) * std
        weights[f"layers.{index}.attn.v_proj.weight"] = torch.randn(config.d_model, config.d_model, device=device) * std
        weights[f"layers.{index}.attn.output_proj.weight"] = torch.randn(config.d_model, config.d_model, device=device) * std
        weights[f"layers.{index}.ln1.weight"] = torch.ones(config.d_model, device=device)
        weights[f"layers.{index}.ffn.w1.weight"] = torch.randn(config.d_ff, config.d_model, device=device) * std
        weights[f"layers.{index}.ffn.w2.weight"] = torch.randn(config.d_model, config.d_ff, device=device) * std
        weights[f"layers.{index}.ffn.w3.weight"] = torch.randn(config.d_ff, config.d_model, device=device) * std
        weights[f"layers.{index}.ln2.weight"] = torch.ones(config.d_model, device=device)

    return weights


def _init_model_from_checkpoint(checkpoint_path: Union[str, Path], config: ModelConfig) -> TransformerLM:
    """
    Loads a checkpoint containing both model weights and hyper-parameters.
    """
    model: TransformerLM = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        weights=_initialize_weights(config=config)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model



def _get_next_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Processes the model's forward pass output into discrete token IDs.
    
    Args:
        logits: Shape (batch, seq_len, vocab_size)
        temperature: Scales logits (higher = more creative/random)
    """
    # We usually only care about the last token predicted in the sequence 
    # for autoregressive generation
    last_token_logits = logits[:, -1, :] / temperature
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

def generate_text(checkpoint_path: Union[str, Path], prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str:
    decoder = tiktoken.get_encoding("gpt2")
    model_config: ModelConfig = ModelConfig(vocab_size=50_257, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, rope_theta=10_000)

    # Initialize the trained model
    model: TransformerLM = _init_model_from_checkpoint(checkpoint_path, model_config)

    # Encode the input prompt
    input_ids: torch.Tensor = torch.tensor(decoder.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate text until max token is reached
    generated: torch.Tensor = input_ids
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = generated[:, -model_config.context_length:]
        
        # Forward pass
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Decode the next token
        next_token = _get_next_token(logits, temperature=temperature)

        # Append token to the sequence
        generated = torch.cat((generated, next_token), dim=1)

    return decoder.decode(generated[0].tolist())


if __name__ == '__main__':
    model_checkpoint = "checkpoint/checkpoint_final.pt"
    prompt = "summarize the story"
    print(generate_text(checkpoint_path=model_checkpoint, prompt=prompt, max_new_tokens=1000))
