"""A tiny dependency-free neural network that learns XOR.

The implementation is intentionally small and readable. It uses one hidden layer,
sigmoid activations, mean squared error, and stochastic gradient descent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def sigmoid(value: float) -> float:
    """Return the sigmoid activation for a single value."""

    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(output: float) -> float:
    """Return the sigmoid derivative using an already-computed sigmoid output."""

    return output * (1.0 - output)


@dataclass
class TinyNeuralNetwork:
    """A small fully connected neural network with one hidden layer."""

    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float = 0.5
    seed: int = 7
    hidden_weights: list[list[float]] = field(init=False)
    output_weights: list[list[float]] = field(init=False)
    hidden_biases: list[float] = field(init=False)
    output_biases: list[float] = field(init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.hidden_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        self.output_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(self.hidden_size)]
            for _ in range(self.output_size)
        ]
        self.hidden_biases = [rng.uniform(-1.0, 1.0) for _ in range(self.hidden_size)]
        self.output_biases = [rng.uniform(-1.0, 1.0) for _ in range(self.output_size)]

    def predict(self, inputs: list[float]) -> list[float]:
        """Run a forward pass and return network outputs."""

        hidden_outputs, final_outputs = self._forward(inputs)
        return final_outputs

    def train(
        self,
        samples: list[tuple[list[float], list[float]]],
        epochs: int = 10_000,
    ) -> float:
        """Train the network and return the final mean squared error."""

        loss = 0.0
        for _ in range(epochs):
            loss = 0.0
            for inputs, expected in samples:
                hidden_outputs, final_outputs = self._forward(inputs)
                loss += self._backpropagate(inputs, hidden_outputs, final_outputs, expected)
        return loss / len(samples)

    def _forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        hidden_outputs = [
            sigmoid(sum(weight * value for weight, value in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.hidden_weights, self.hidden_biases)
        ]
        final_outputs = [
            sigmoid(
                sum(weight * value for weight, value in zip(weights, hidden_outputs))
                + bias
            )
            for weights, bias in zip(self.output_weights, self.output_biases)
        ]
        return hidden_outputs, final_outputs

    def _backpropagate(
        self,
        inputs: list[float],
        hidden_outputs: list[float],
        final_outputs: list[float],
        expected: list[float],
    ) -> float:
        output_deltas = [
            (target - output) * sigmoid_derivative(output)
            for target, output in zip(expected, final_outputs)
        ]
        hidden_deltas = []
        for hidden_index, hidden_output in enumerate(hidden_outputs):
            downstream_error = sum(
                output_deltas[output_index]
                * self.output_weights[output_index][hidden_index]
                for output_index in range(self.output_size)
            )
            hidden_deltas.append(downstream_error * sigmoid_derivative(hidden_output))

        for output_index in range(self.output_size):
            for hidden_index, hidden_output in enumerate(hidden_outputs):
                self.output_weights[output_index][hidden_index] += (
                    self.learning_rate * output_deltas[output_index] * hidden_output
                )
            self.output_biases[output_index] += self.learning_rate * output_deltas[output_index]

        for hidden_index in range(self.hidden_size):
            for input_index, input_value in enumerate(inputs):
                self.hidden_weights[hidden_index][input_index] += (
                    self.learning_rate * hidden_deltas[hidden_index] * input_value
                )
            self.hidden_biases[hidden_index] += self.learning_rate * hidden_deltas[hidden_index]

        return sum((target - output) ** 2 for target, output in zip(expected, final_outputs))


def main() -> None:
    samples = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    network = TinyNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    loss = network.train(samples)

    print(f"Final loss: {loss:.6f}")
    for inputs, expected in samples:
        prediction = network.predict(inputs)[0]
        print(f"{inputs} -> {prediction:.3f} (expected {expected[0]:.0f})")


if __name__ == "__main__":
    main()
