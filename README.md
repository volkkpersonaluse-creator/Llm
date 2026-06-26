# Tiny Neural Network and BigLLM

This repository contains three minimal, dependency-free AI examples implemented in Python.

- `tiny_neural_network.py` trains a small multilayer perceptron on the XOR problem using backpropagation.
- `big_llm.py` trains `BigLLM`, a compact character-level language model that learns next-character probabilities from a tiny built-in corpus and generates text.
- `advanced_neural_network.py` trains a deeper multilayer perceptron on a noisy two-spiral classification task, which requires a curved non-linear decision boundary.

These examples are educational and intentionally small enough to inspect in one sitting. `BigLLM` is not a production-scale large language model; it is a local toy model that demonstrates the train/generate loop used by language models.

## Run the neural network

```bash
python tiny_neural_network.py
```

The script prints the final loss and predicted outputs for the four XOR inputs.

## Train and run BigLLM

```bash
python big_llm.py
```

The script trains the character model on its built-in corpus, prints `Training complete.`, and generates a short text sample from the prompt `A language`.


## Train the deeper neural network

```bash
python advanced_neural_network.py
```

The script generates a noisy two-spiral dataset, trains a `[2, 12, 8, 1]` neural network, and prints final loss plus training accuracy.
