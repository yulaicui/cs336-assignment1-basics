import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiktoken


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


if __name__ == '__main__':
    training_dataset = _encode_txt(Path("data/owt_train.txt"))
    validation_dataset = _encode_txt(Path("data/owt_valid.txt"))