"""A deeper dependency-free neural network for a harder classification task.

The demo trains a multilayer perceptron on a noisy two-spiral dataset. Two spirals
are a classic non-linear classification problem that is much harder than XOR
because the decision boundary must curve around the input space many times.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)


@dataclass
class DeepNeuralNetwork:
    """A fully connected neural network with any number of hidden layers."""

    layer_sizes: list[int]
    learning_rate: float = 0.35
    seed: int = 19
    weights: list[list[list[float]]] = field(init=False)
    biases: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output layers")

        rng = random.Random(self.seed)
        self.weights = []
        self.biases = []
        for input_size, output_size in zip(self.layer_sizes, self.layer_sizes[1:]):
            limit = math.sqrt(6.0 / (input_size + output_size))
            self.weights.append(
                [
                    [rng.uniform(-limit, limit) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            )
            self.biases.append([0.0 for _ in range(output_size)])

    def predict(self, inputs: list[float]) -> list[float]:
        """Return model outputs for one input row."""

        return self._forward(inputs)[-1]

    def train(
        self,
        samples: list[tuple[list[float], list[float]]],
        epochs: int = 700,
    ) -> float:
        """Train with stochastic gradient descent and return final MSE."""

        rng = random.Random(self.seed + 1)
        loss = 0.0
        for _ in range(epochs):
            rng.shuffle(samples)
            loss = 0.0
            for inputs, expected in samples:
                activations = self._forward(inputs)
                loss += self._backpropagate(activations, expected)
        return loss / len(samples)

    def _forward(self, inputs: list[float]) -> list[list[float]]:
        activations = [inputs]
        current = inputs
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            current = [
                sigmoid(sum(weight * value for weight, value in zip(row, current)) + bias)
                for row, bias in zip(layer_weights, layer_biases)
            ]
            activations.append(current)
        return activations

    def _backpropagate(self, activations: list[list[float]], expected: list[float]) -> float:
        output = activations[-1]
        deltas = [
            [(target - value) * sigmoid_derivative(value) for target, value in zip(expected, output)]
        ]

        for layer_index in range(len(self.weights) - 2, -1, -1):
            next_weights = self.weights[layer_index + 1]
            next_deltas = deltas[0]
            layer_delta = []
            for neuron_index, activation in enumerate(activations[layer_index + 1]):
                downstream_error = sum(
                    next_deltas[next_index] * next_weights[next_index][neuron_index]
                    for next_index in range(len(next_deltas))
                )
                layer_delta.append(downstream_error * sigmoid_derivative(activation))
            deltas.insert(0, layer_delta)

        for layer_index, layer_delta in enumerate(deltas):
            previous_activation = activations[layer_index]
            for neuron_index, delta in enumerate(layer_delta):
                for weight_index, activation in enumerate(previous_activation):
                    self.weights[layer_index][neuron_index][weight_index] += (
                        self.learning_rate * delta * activation
                    )
                self.biases[layer_index][neuron_index] += self.learning_rate * delta

        return sum((target - value) ** 2 for target, value in zip(expected, output))


def make_spiral_dataset(points_per_class: int = 70, noise: float = 0.18) -> list[tuple[list[float], list[float]]]:
    """Create a small two-spiral classification dataset."""

    rng = random.Random(5)
    samples = []
    for label in (0, 1):
        for index in range(points_per_class):
            radius = index / points_per_class
            angle = 1.75 * math.pi * radius + label * math.pi
            x = radius * math.cos(angle) + rng.uniform(-noise, noise) * 0.2
            y = radius * math.sin(angle) + rng.uniform(-noise, noise) * 0.2
            samples.append(([0.5 + x / 2.0, 0.5 + y / 2.0], [float(label)]))
    return samples


def accuracy(model: DeepNeuralNetwork, samples: list[tuple[list[float], list[float]]]) -> float:
    correct = 0
    for inputs, expected in samples:
        predicted = 1.0 if model.predict(inputs)[0] >= 0.5 else 0.0
        if predicted == expected[0]:
            correct += 1
    return correct / len(samples)


def main() -> None:
    samples = make_spiral_dataset()
    model = DeepNeuralNetwork(layer_sizes=[2, 12, 8, 1])
    loss = model.train(samples)
    print(f"Final loss: {loss:.6f}")
    print(f"Training accuracy: {accuracy(model, samples):.1%}")


if __name__ == "__main__":
    main()
