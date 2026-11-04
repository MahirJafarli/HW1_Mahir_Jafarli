import json

class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int) -> None:
        """Trains the BPE tokenizer on raw text up to the specified vocab_size."""
        # A1: Encode text to UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        
        # Base vocabulary of 256 byte values
        num_merges = vocab_size - 256
        self.merges = []

        for i in range(num_merges):
            # Count adjacent pairs
            stats = {}
            for p1, p2 in zip(tokens[:-1], tokens[1:]):
                stats[(p1, p2)] = stats.get((p1, p2), 0) + 1
            
            if not stats:
                break
                
            # A4: Tie-breaking rule: pick most frequent pair, smallest a, then smallest b
            best_pair = max(stats.keys(), key=lambda p: (stats[p], -p[0], -p[1]))
            
            # A3: New ID for the merge
            new_id = 256 + i
            
            # Replace occurrences left to right
            tokens = self._merge_pair(tokens, best_pair, new_id)
            self.merges.append(best_pair)

        # Build final decoding vocabulary mapping IDs to byte sequences
        self._build_vocab()

    def _merge_pair(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replaces non-overlapping occurrences of a pair with new_id left to right."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _build_vocab(self):
        """Constructs the mapping from token IDs back to bytes."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for idx, (p1, p2) in enumerate(self.merges):
            new_id = 256 + idx
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]

    def encode(self, text: str) -> list[int]:
        """Applies learned merges in the exact order they were learned."""
        tokens = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            new_id = 256 + i
            tokens = self._merge_pair(tokens, pair, new_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a UTF-8 string."""
        byte_arr = b"".join(self.vocab[idx] for idx in ids)
        return byte_arr.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Saves merges to a JSON file."""
        data = {"merges": [[p1, p2] for p1, p2 in self.merges]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Loads tokenizer merges from a JSON file."""
        tokenizer = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._build_vocab()
        return tokenizer