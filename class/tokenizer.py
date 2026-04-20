from typing import Iterable, Iterator
import regex as re
import json
class Tokenizer:
    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        A class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        vocab = {}
        merges = []

        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            raw_vocab = json.load(vf)
            for k, v in raw_vocab.items():
                vocab[int(k)] = eval(v)

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                left_str, right_str = line.split()
                merges.append((eval(left_str), eval(right_str)))

        return cls(vocab, merges, special_tokens)
        
        
    def encode(self, text: str) -> list[int]:
        """
        Encode a given string of text into a list of token IDs. 
        This method should apply the BPE merges to the input text
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ids = []

        # Step 1: split by special tokens first, so they stay atomic
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)#example: text = "helloworld"special_tokens = ["world"]
            # special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens) 
            parts = re.split(f"({special_pattern})", text) # example: ["hello", "<|endoftext|>", "world"]
            
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # Step 2: if this part is a special token, map directly to id
            if part in self.special_tokens:
                ids.append(self.token_to_id[part.encode("utf-8")])
                continue

            # Step 3: pre-tokenize normal text
            pieces = re.findall(PAT, part)

            for piece in pieces:
                # convert one piece into byte-level tokens
                token_seq = tuple(bytes([b]) for b in piece.encode("utf-8")) #example: "hello" -> (b'h', b'e', b'l', b'l', b'o')

                # Step 4: repeatedly apply the highest-priority merge
                while len(token_seq) >= 2:
                    best_pair = None
                    best_rank = float("inf")

                    for i in range(len(token_seq) - 1):
                        pair = (token_seq[i], token_seq[i + 1])
                        if pair in self.merge_ranks and self.merge_ranks[pair] < best_rank:
                            best_pair = pair
                            best_rank = self.merge_ranks[pair]

                    if best_pair is None:
                        break

                    new_token = best_pair[0] + best_pair[1]
                    new_seq = []
                    i = 0

                    while i < len(token_seq):
                        if (
                            i < len(token_seq) - 1
                            and token_seq[i] == best_pair[0]
                            and token_seq[i + 1] == best_pair[1]
                        ):
                            new_seq.append(new_token)
                            i += 2
                        else:
                            new_seq.append(token_seq[i])
                            i += 1

                    token_seq = tuple(new_seq)

                # Step 5: map merged byte tokens to vocab ids
                for token in token_seq:
                    ids.append(self.token_to_id[token])

        return ids
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        To test your Tokenizer against our provided tests, you will first need to implement the test
        adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py. Your
        implementation should be able to pass all tests.
        """
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
    
