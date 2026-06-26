"""A small, trainable character language model named BigLLM.

This is not a production-scale LLM. It is a dependency-free teaching example that
learns next-character probabilities from a tiny corpus and generates text from
those learned transitions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import random


class BigLLM:
    """A compact character-level n-gram language model."""

    def __init__(self, order: int = 4, seed: int = 11) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self._rng = random.Random(seed)
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self._fallback: Counter[str] = Counter()

    def train(self, text: str) -> None:
        """Learn next-character counts from text."""

        if len(text) <= self.order:
            raise ValueError("training text must be longer than the model order")

        padded = "~" * self.order + text
        for index in range(len(padded) - self.order):
            context = padded[index : index + self.order]
            next_character = padded[index + self.order]
            self._transitions[context][next_character] += 1
            self._fallback[next_character] += 1

    def generate(self, prompt: str = "", length: int = 160) -> str:
        """Generate text by sampling learned next-character probabilities."""

        if not self._transitions:
            raise ValueError("train the model before generating text")
        if length < 1:
            return prompt

        generated = prompt
        context = ("~" * self.order + prompt)[-self.order :]
        for _ in range(length):
            choices = self._transitions.get(context, self._fallback)
            next_character = self._sample(choices)
            generated += next_character
            context = (context + next_character)[-self.order :]
        return generated

    def _sample(self, counts: Counter[str]) -> str:
        total = sum(counts.values())
        pick = self._rng.uniform(0, total)
        current = 0.0
        for character, count in counts.items():
            current += count
            if current >= pick:
                return character
        return counts.most_common(1)[0][0]


TRAINING_CORPUS = """
A language model learns patterns in text.
This tiny ai studies characters, remembers local context, and writes new lines.
Neural networks can learn from examples, while language models predict what comes next.
Small demos are useful because they make training easy to inspect.
The model is named BigLLM for fun, but it is intentionally compact and educational.
""".strip()


def main() -> None:
    model = BigLLM(order=5)
    model.train(TRAINING_CORPUS)
    print("Training complete.")
    print(model.generate(prompt="A language", length=180))


if __name__ == "__main__":
    main()
