import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiktoken
import torch

from cs336_basics.config import ModelConfig, TrainingConfig, ValidationConfig
from cs336_basics.nn_modules import TransformerLM
from cs336_basics.training import AdamW, cross_entropy, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _initialize_weights(config: ModelConfig) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    weights['token_embeddings.weight'] = torch.empty(config.vocab_size, config.d_model, device=device)
    weights['ln_final.weight'] = torch.ones(config.d_model, device=device)
    weights['lm_head.weight'] = torch.empty(config.d_model, config.vocab_size, device=device)

    for index in range(config.num_layers):
        weights[f"layers.{index}.attn.q_proj.weight"] = torch.empty(config.d_model, config.d_model, device=device)
        weights[f"layers.{index}.attn.k_proj.weight"] = torch.empty(config.d_model, config.d_model, device=device)
        weights[f"layers.{index}.attn.v_proj.weight"] = torch.empty(config.d_model, config.d_model, device=device)
        weights[f"layers.{index}.attn.output_proj.weight"] = torch.empty(config.d_model, config.d_model, device=device)
        weights[f"layers.{index}.ln1.weight"] = torch.ones(config.d_model, device=device)
        weights[f"layers.{index}.ffn.w1.weight"] = torch.empty(config.d_ff, config.d_model, device=device)
        weights[f"layers.{index}.ffn.w2.weight"] = torch.empty(config.d_model, config.d_ff, device=device)
        weights[f"layers.{index}.ffn.w3.weight"] = torch.empty(config.d_ff, config.d_model, device=device)
        weights[f"layers.{index}.ln2.weight"] = torch.ones(config.d_model, device=device)

    return weights


def _encode_txt(txt_file_path: Path) -> npt.NDArray:
    print(f"encoding text file: {txt_file_path}")
    assert txt_file_path.suffix == ".txt", f"expecting a .txt file, got {txt_file_path} instead"
    
    encoded_bin_path: Path = txt_file_path.with_suffix(".bin")
    if not encoded_bin_path.exists():
        encoding = tiktoken.get_encoding("gpt2")
        with open(encoded_bin_path, "wb") as f_out:
            with open(txt_file_path, "r", encoding="utf-8") as f_in:
                line_count = 0
                for line in f_in:
                    line_tokens = encoding.encode_ordinary(line)
                    f_out.write(np.array(line_tokens, dtype=np.uint32).tobytes())
                    line_count += 1

                    if line_count % 1000_000 == 0:
                        print(f"encoded {line_count} lines")

    num_tokens: int = os.path.getsize(str(encoded_bin_path)) // 4
    print(f"found {num_tokens} encoded tokens from {encoded_bin_path}")
    return np.memmap(str(encoded_bin_path), dtype=np.uint32, mode='r', shape=(num_tokens,))


def train_and_evaluate(
        model_config: ModelConfig, 
        training_config: TrainingConfig,
        validation_config: ValidationConfig,
        train_dataset: npt.NDArray,
        val_dataset: npt.NDArray,
    ):
    initial_weights: dict[str, torch.Tensor] = _initialize_weights(config=model_config)

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
        weights=initial_weights
    )

    optimizer = AdamW(params=model.parameters())

    for epoc in training_config.epocs:
        model.train()
        x, y = load_data(
            train_dataset, 
            training_config.batch_size, 
            training_config.context_length, 
            device.type, 
            training_config.random_seed
        )
        
        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass and optimize gradients
        loss.backward()
        optimizer.step() # update weights based on gradients
        optimizer.zero_grad(set_to_none=True) # wipes out gradients, set to None to release memory allocation

        if epoc > 0 and epoc % validation_config.val_every == 0:
            model.eval()
            val_loss = 0.0
            val_iter_num = 0

            for val_x, val_y in load_data(val_dataset, validation_config.batch_size, validation_config.context_length, device.type, validation_config.random_seed):
                val_logits = model(val_x)
                val_loss = cross_entropy(val_logits, val_y)
                val_loss += val_loss.item()
                val_iter_num += 1
            
            print(f"Epoc: {epoc} | validation loss: {val_loss / val_iter_num:.4f}")


# if __name__ == '__main__':
#     training_dataset = _encode_txt(Path("data/TinyStoriesV2-GPT4-train.txt"))
#     validation_dataset = _encode_txt(Path("data/TinyStoriesV2-GPT4-valid.txt"))
