import os
import regex as re
from collections import defaultdict
from typing import Iterable, Iterator, Protocol
from tqdm.contrib.concurrent import process_map


# reference: https://github.com/openai/tiktoken/pull/234/files
GPT2_PRE_TOKENIZER_REGEX = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def split_text_to_words(text: str) -> list[str]:
    return GPT2_PRE_TOKENIZER_REGEX.findall(text)


def split_text_to_words_iterator(text: str) -> Iterator[str]:
    return GPT2_PRE_TOKENIZER_REGEX.finditer(text)


def word_to_bytes(word: str) -> list[bytes]:
    return list(bytes([b]) for b in word.encode("utf-8"))


def split_string_by_special_tokens(string: str, special_tokens: list[str]) -> list[str]:
    """
    Split a full text into segments separated by special tokens.
    Special tokens themselves are treated as segments too.

    Example:
        imagine <|endoftext|> as a special token (that we don't want to merge with other tokens),
        running this method on "Héllò hôw <|endoftext|>are ü? 🙃<|endoftext|>" should give us
        ["Héllò hôw ", "<|endoftext|>", "are ü? 🙃", "<|endoftext|>"]

    Args:
        string (str): the original text string.

    Returns:
        list[str]: the string segmented by split by special tokens.
    """
    if not special_tokens:
        return [string]

    pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
    return [c for c in re.compile( f"({pattern})").split(string) if c]  # remove empty strings


def get_word_counts_of_chunk(chunk: str) -> dict[str, int]:
    word_counts: dict[str, int] = defaultdict(int)
    word_matches = split_text_to_words_iterator(chunk)
    for match in word_matches:
        word = match.group(0)
        word_counts[word] += 1
    return word_counts


def get_global_word_counts(word_counts_of_chunks: list[dict[str, int]]) -> dict[tuple, int]:
    word_counts = defaultdict(int)
    for workd_counts_of_chunk in word_counts_of_chunks:
        for word, count in workd_counts_of_chunk.items():
            word_counts[tuple(word_to_bytes(word))] += count
    return word_counts


def merge_token_strings(token_strings: list[str], merged_string: str) -> list[str]:
    new_token_strings: list[str] = []
    i = 0
    while i < len(token_strings):
        if i < len(token_strings) - 1 and token_strings[i] + token_strings[i+1] == merged_string:
            new_token_strings.append(merged_string)
            i += 2
        else:
            new_token_strings.append(token_strings[i])
            i += 1
    
    return new_token_strings


class Tokenizer(Protocol):
    def encode(self, string: str) -> list[int]:
        """
        Encode a string into tokens.

        Args:
            string (str): the orginal string.

        Returns:
            list[int]: tokens encoded from the original string.
        """
        ...
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode a string into tokens, optimized for memory efficiency by yielding as a generator.

        Args:
            iterable (Iterable[str]): the orginal string.

        Yields:
            Iterator[int]: tokens encoded from the original string.
        """
        ...
    
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of tokens to a string.

        Args:
            indices (list[int]): tokens to decode.

        Returns:
            str: decoded string.
        """
        ...


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE by generating vocabulary and bytes to merge from a text file.

    Args:
        input_path (str | os.PathLike): input file.
        vocab_size (int): number of tokens in vocabulary
        special_tokens (list[str]): tokens not considered for BPE training (i.e. <|endoftext|>)

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: vocab and merges
    """
    # initialize vocab as all 256 bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # include special tokens to vocab
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    # compute merges from strings read from input file
    with open(input_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        chunks = split_string_by_special_tokens(file_content, special_tokens)
        chunks = [chunk for chunk in chunks if chunk not in set(special_tokens)]

        # count words
        if len(chunks) < 4: word_counts_of_chunks = list(map(get_word_counts_of_chunk, chunks))
        else: word_counts_of_chunks = process_map(get_word_counts_of_chunk, chunks)
        word_counts: dict[tuple, int] = get_global_word_counts(word_counts_of_chunks)
        
        # get initial pair counds from pre-tokenized word counts
        byte_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for word, word_count in word_counts.items():
            for l_bytes, r_bytes in zip(word, word[1:]):
                byte_pair_counts[(l_bytes, r_bytes)] += word_count
        
        # find merges until vocab size is reached
        merges: list[tuple[bytes, bytes]] = []
        merge_iterations: int = vocab_size - len(vocab)
        for _ in range(merge_iterations):
            # find the pair to merge, update merges and vocab accordingly
            pair_to_merge: tuple[bytes, bytes] = max(byte_pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            merged_pair = pair_to_merge[0] + pair_to_merge[1]
            vocab[len(vocab)] = merged_pair
            merges.append(pair_to_merge)

            # update pair counts
            new_word_counts: dict[tuple, int] = defaultdict(int)
            for word_bytes, word_count in word_counts.items():
                old_byte_pairs = list(zip(word_bytes, word_bytes[1:]))
                if pair_to_merge in old_byte_pairs:
                    # decrement byte pair counts from words that need merging
                    for l_bytes, r_bytes in old_byte_pairs:
                        byte_pair_counts[(l_bytes, r_bytes)] -= word_count
                    
                    # merge bytes
                    i = 0
                    merged_word_bytes: list[bytes] = []
                    while i < len(word_bytes):
                        if i + 1 < len(word_bytes) and word_bytes[i] + word_bytes[i+1] == merged_pair:
                            merged_word_bytes.append(merged_pair)
                            i += 2
                        else:
                            merged_word_bytes.append(word_bytes[i])
                            i += 1

                    # re-increment pair counts
                    for l_bytes, r_bytes in zip(merged_word_bytes, merged_word_bytes[1:]):
                        byte_pair_counts[(l_bytes, r_bytes)] += word_count
                    
                    # update word counts
                    new_word_counts[tuple(merged_word_bytes)] = word_count
                else:
                    new_word_counts[word_bytes] = word_count
            word_counts = new_word_counts
        
        return vocab, merges


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding Tokenizer.
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self._vocab: dict[int, bytes] = vocab # index -> bytes
        self._merges: set[tuple[bytes, bytes]] = set(merges)  # (byte_1, byte_2) to merge
        self._special_tokens: list[str] = sorted(special_tokens, key=len, reverse=True) if special_tokens else [] # special tokens from longest to shortest
        self._bytes_to_token: dict[bytes, int] = {byte: index for index, byte in vocab.items()} # bytes -> index reverse look-up

    def _merge_bytes(self, string: str) -> list[bytes]:
        """
        Convert a text string into bytes that are recognized by BPE vocabulary.

        Args:
            string (_type_): the text string.

        Returns:
            list[bytes]: merged bytes.
        """
        merged_bytes_list: list[bytes] = []
        word_list: list[str] = split_text_to_words(string) # split the string into pre-tokenized words
        
        for word in word_list:
            merged_word_bytes = word_to_bytes(word)
            while True:
                min_merged_token = float('inf')
                index_to_merge = -1
                merged_bytes = None
                for i in range(len(merged_word_bytes) - 1):
                    if (merged_word_bytes[i], merged_word_bytes[i+1]) in self._merges:
                        merged_token = self._bytes_to_token[merged_word_bytes[i] + merged_word_bytes[i+1]]
                        if merged_token < min_merged_token: # prefer smaller merged token (for tie-breaking)
                            min_merged_token = merged_token
                            index_to_merge = i
                            merged_bytes = merged_word_bytes[i] + merged_word_bytes[i+1]
                
                if index_to_merge != -1: # if some tokens are merged during the iteration, we update the bytes list
                    merged_word_bytes = merged_word_bytes[:index_to_merge] + [merged_bytes] + merged_word_bytes[index_to_merge + 2:]
                else:
                    break
            
            merged_bytes_list.extend(merged_word_bytes)

        return merged_bytes_list

    def encode(self, string: str) -> list[int]:
        segments: list[str] = split_string_by_special_tokens(string, self._special_tokens)
        merged_tokens: list[int] = []
        for segment in segments:
            if segment in set(self._special_tokens):
                merged_tokens.append(self._bytes_to_token[segment.encode('utf-8')])
            else:
                merged_segment_bytes = self._merge_bytes(segment)
                merged_tokens.extend([self._bytes_to_token[token_bytes] for token_bytes in merged_segment_bytes])

        return merged_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[str]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, tokens: list[int]) -> str:
        bytes_list = [self._vocab[token] for token in tokens]
        return b"".join(bytes_list).decode("utf-8", errors='replace')
