import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class BytePairEncoder:
    """
    A simple Byte-Pair Encoding (BPE) implementation.
    Starts with an alphanumeric vocabulary and iteratively merges
    the most frequent byte pairs until the desired vocab size is reached.

    Args:
        vocab (Optional[Dict[str, int]]): A dictionary mapping tokens to their indices.
        Default is a basic alphanumeric vocabulary.
        merges (Optional[List[Tuple[str, str]]]): A list of tuples representing the merges to be applied.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
    ):
        if vocab is None:
            self.vocab = {
                c: i
                for i, c in enumerate(
                    list(
                        "abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                    )
                    + [" "]
                )
            }
        else:
            self.vocab = vocab
        self.merges = merges or []

    def _get_stats(self, corpus: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Count the frequency of each pair of adjacent tokens in the corpus.
        Returns a dictionary mapping pairs to their frequencies.
        """
        pairs = Counter()
        for token_list in corpus:
            for i in range(len(token_list) - 1):
                pairs[(token_list[i], token_list[i + 1])] += 1
        return pairs

    def _merge_pair(
        self, pair: Tuple[str, str], corpus: List[List[str]]
    ) -> List[List[str]]:
        """
        Merge the most frequent pair in the corpus.
        Returns a new corpus with the merged pair.
        """
        pattern = re.escape(" ".join(pair))
        repl = "".join(pair)
        merged_corpus = []
        for token_list in corpus:
            token_str = " ".join(token_list)
            # merge all occurrences of the pair
            merged = re.sub(pattern, repl, token_str)
            merged_corpus.append(merged.split(" "))
        return merged_corpus

    def fit(self, text: str, target_vocab_size: int):
        """
        Learn merges from the provided text until the vocab size
        reaches target_vocab_size.
        """
        # initialize corpus as list of token lists
        corpus = [[c for c in word] for word in text.split()]
        logger.info("Starting BPE training. Initial vocab size: %d", len(self.vocab))

        while len(self.vocab) < target_vocab_size:
            stats = self._get_stats(corpus)
            if not stats:
                break
            best_pair, freq = max(stats.items(), key=lambda x: x[1])
            if freq < 2:
                logger.info("No pair occurs more than once. Stopping.")
                break
            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            self.vocab[merged_token] = len(self.vocab)
            corpus = self._merge_pair(best_pair, corpus)
            logger.info(
                "Merged pair %s -> %s. New vocab size: %d",
                best_pair,
                merged_token,
                len(self.vocab),
            )

        logger.info("BPE training complete. Final vocab size: %d", len(self.vocab))

    def fine_tune(self, text: str, target_vocab_size: int):
        """
        Fine‑tune the existing BPE model by learning additional merges
        from `text` until vocab size reaches `target_vocab_size`.
        Applies existing merges on the new corpus before further training.
        """
        if target_vocab_size <= len(self.vocab):
            logger.info(
                "Target vocab size (%d) ≤ current vocab size (%d); skipping fine‑tune.",
                target_vocab_size,
                len(self.vocab),
            )
            return

        # build fresh corpus
        corpus = [[c for c in word] for word in text.split()]

        logger.info("Merging existing pairs into the corpus.")
        # apply all existing merges to the corpus
        for i, merge in enumerate(self.merges):
            corpus = self._merge_pair(merge, corpus)
            if i % 100 == 0:
                logger.info("Applied %d merges to the corpus.", i)

        logger.info("Starting BPE fine‑tuning. Current vocab size: %d", len(self.vocab))
        # continue merging until target vocab size
        while len(self.vocab) < target_vocab_size:
            stats = self._get_stats(corpus)
            if not stats:
                break
            best_pair, freq = max(stats.items(), key=lambda x: x[1])
            if freq < 2:
                logger.info("No pair occurs more than once. Stopping fine‑tune.")
                break

            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            self.vocab[merged_token] = len(self.vocab)
            corpus = self._merge_pair(best_pair, corpus)
            logger.info(
                "Fine‑tuned merge %s -> %s. New vocab size: %d",
                best_pair,
                merged_token,
                len(self.vocab),
            )

        logger.info("BPE fine‑tuning complete. Final vocab size: %d", len(self.vocab))

    def encode(self, text: str) -> List[int]:
        """
        Tokenize and encode text to a sequence of vocabulary indices.
        """
        tokens: List[int] = []
        for word in text.split():
            symbols = list(word)
            for merge in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                        symbols[i : i + 2] = ["".join(merge)]
                    else:
                        i += 1
            for sym in symbols:
                idx = self.vocab.get(sym)
                if idx is None:
                    raise ValueError(f"Symbol '{sym}' not in vocabulary.")
                tokens.append(idx)
            # preserve spaces if in vocab
            if " " in self.vocab:
                tokens.append(self.vocab[" "])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a sequence of vocabulary indices back to text.
        """
        inv_vocab = {i: tok for tok, i in self.vocab.items()}
        text = "".join(inv_vocab[idx] for idx in tokens)
        return text.strip()

    def save(self, filepath: str):
        """
        Save the vocabulary and merges to a JSON file.
        """
        out = {"vocab": self.vocab, "merges": [" ".join(pair) for pair in self.merges]}
        Path(filepath).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        logger.info("Saved BPE model to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "BytePairEncoder":
        """
        Load a BPE model from a JSON file.
        """
        obj = json.loads(Path(filepath).read_text())
        vocab = {tok: idx for tok, idx in obj["vocab"].items()}
        merges = [tuple(pair.split(" ")) for pair in obj["merges"]]
        logger.info("Loaded BPE model from %s", filepath)
        return cls(vocab=vocab, merges=merges)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or use a BPE tokenizer.")
    parser.add_argument("--train", type=str, help="Path to training text file.")
    parser.add_argument(
        "--vocab-size", type=int, default=1000, help="Target vocabulary size."
    )
    parser.add_argument(
        "--save",
        type=str,
        default="bpe_model.json",
        help="Path to save the trained BPE model.",
    )
    parser.add_argument("--encode", type=str, help="Text to encode with loaded model.")
    parser.add_argument("--model", type=str, help="Path to an existing BPE model JSON.")
    args = parser.parse_args()

    if args.model and args.encode:
        encoder = BytePairEncoder.load(args.model)
        print(encoder.encode(args.encode))
    elif args.train:
        text = Path(args.train).read_text(encoding="utf-8")
        encoder = BytePairEncoder()
        encoder.fit(text, args.vocab_size)
        encoder.save(args.save)
    else:
        parser.print_help()
