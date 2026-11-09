# How Does a Neural Network Actually Learn?

## Backpropagation and Gradient Descent Built From Scratch

The notebook opens up the part of neural-network training that is usually hidden behind calls such as `loss.backward()` and `optimizer.step()`.

Instead of using PyTorch or TensorFlow to implement the learning mechanism, the notebook builds a small scalar automatic-differentiation engine in plain Python. That engine is then used to train a multi-layer perceptron on a non-linear classification problem.

The main notebook is [HowNeuralNetLearn.ipynb](HowNeuralNetLearn.ipynb).

## What This Tutorial Teaches

By the end of the notebook, you should be able to explain and demonstrate:

- What a derivative and a partial derivative measure.
- How the gradient describes the direction in which a loss increases.
- Why gradient descent updates parameters in the opposite direction of the gradient.
- How a computation graph decomposes a complicated expression into simple operations.
- How the chain rule becomes backpropagation when applied from the output toward the inputs.
- Why gradients must be accumulated across multiple computation paths.
- Why gradients must be reset before every training step.
- How an automatic-differentiation engine can construct and traverse a computation graph.
- How a neural network learns a curved decision boundary from data.
- Why finite differences are useful for checking custom gradient implementations.
- How learning rate, activation saturation, and generalisation affect training.

The tutorial is designed for readers who know basic Python, classes and closures, introductory machine-learning terminology, and the derivative from first-year calculus. No previous PyTorch or TensorFlow experience is required.

## Requirements

The notebook uses:

- Python 3
- NumPy
- Matplotlib
- scikit-learn

PyTorch is optional. One final cell uses it only as an additional comparison if it is installed; the notebook catches `ImportError` and continues without it.

For example, the required packages can be installed with:

```bash
python -m pip install numpy matplotlib scikit-learn jupyter
```

Then open the notebook in Jupyter or VS Code and run the cells from top to bottom.

## Notebook Roadmap

### 1. Setup and reproducibility

The first code cell imports the numerical and plotting libraries and fixes the Python and NumPy random seeds at `42`. The seed makes the neural-network initialisation and generated dataset reproducible. Plot dimensions, grid visibility, and grid transparency are also configured here.

### 2. Why the gradient matters

The notebook first uses a one-parameter model:

```text
y_hat = w * x
```

The toy data follows `y = 2x`, so the ideal value of `w` is `2`. A mean-squared-error curve is plotted for many values of `w`, and a numerical slope is estimated by making a small change to `w`.

This example establishes the central rule:

```text
parameter <- parameter - learning_rate * gradient
```

A negative slope means that increasing the parameter reduces the loss. A positive slope means that decreasing it reduces the loss. The notebook also explains why estimating every gradient by perturbing one parameter at a time is infeasible for a modern network.

### 3. Derivatives, computation graphs, and the chain rule

The notebook defines a finite-difference estimate of a derivative and compares it with the exact derivative of:

```text
f(x) = 3x^2 - 4x + 5
```

As the step size becomes smaller, the finite-difference estimate approaches the exact derivative.

It then works through the computation graph:

```text
L = (x * w + b)^2
```

with `x = 2`, `w = -3`, and `b = 10`. The forward calculation gives `L = 16`. Applying the chain rule backward gives:

```text
dL/dx = -24
dL/dw = 16
dL/db = 8
```

The notebook verifies those values numerically. This is an important distinction: backpropagation is not a different kind of calculus; it is an efficient organisation of the chain rule over a computation graph.

### 4. The `Value` automatic-differentiation engine

The `Value` class stores both the numerical result of an operation and enough information to differentiate it later:

- `data`: the forward-pass scalar value.
- `grad`: the derivative of the final loss with respect to that value.
- `_prev`: the parent values that produced it.
- `_backward`: a function containing the local derivative rule.
- `_op`: a label describing the operation.

The engine implements addition, multiplication, powers, negation, subtraction, division, and `tanh`. Every operation creates a new `Value` and attaches a backward function. During `backward()`, the graph is topologically sorted so that each node receives all downstream gradient contributions before passing its gradient to its parents.

Two implementation details are especially important:

1. Gradients use `+=` because a value can influence the output through more than one path. Those path contributions must be added.
2. `backward()` does not clear old gradients. The training loop therefore explicitly zeros parameter gradients before each backward pass.

The engine reproduces the hand-calculated gradients for the expression from the previous section without containing any formula-specific derivative code.

### 5. Gradient descent on the toy model

The one-parameter example is trained from an initial value of `w = 0.5`. Each iteration performs the complete training sequence:

1. Build the prediction and loss in a forward pass.
2. Reset the parameter gradient.
3. Run `loss.backward()`.
4. Update `w` using gradient descent.

The loss should decrease and `w` should approach the true value `2.0`. This small example makes the mechanics of a neural-network training step visible before the notebook introduces many parameters.

### 6. Training a neural network on `make_moons`

The worked example uses scikit-learn's `make_moons` dataset. The two classes form interleaving half-circles, so a straight-line model cannot separate them well. The labels are converted from `0` and `1` to `-1` and `+1` so they match the output range of `tanh`.

The network is built from the custom `Value` objects:

```text
2 inputs -> 16 hidden units -> 16 hidden units -> 1 output
```

Each neuron computes:

```text
output = tanh(sum(weight_i * input_i) + bias)
```

The model has 337 trainable parameters:

```text
Layer 1: 2 * 16 + 16       = 48
Layer 2: 16 * 16 + 16      = 272
Output:  16 * 1 + 1        = 17
Total                         337
```

The loss is mean squared error over the 100 training examples:

```text
loss = mean((prediction - label)^2)
```

MSE is used to keep the custom engine small. A production binary classifier would commonly use a cross-entropy loss, but the backpropagation mechanism would remain the same.

Before training, the notebook checks selected parameter gradients using centred finite differences. Backpropagation computes all 337 gradients in one backward pass, while a finite-difference check for one parameter requires two additional forward evaluations of the dataset.

The training loop uses a learning rate of `0.1` for `150` steps. It records both loss and training accuracy, then evaluates the trained model on 300 fresh test points. A contour plot displays the learned output surface and the decision boundary at output `0`.

The expected qualitative result is a smooth, curved boundary between the moons. The test evaluation is included to distinguish learning a useful boundary from merely memorising the 100 training examples.

### 7. Optional PyTorch comparison

If PyTorch is available, the notebook evaluates the same scalar expression in PyTorch and compares its gradients with the custom engine:

```text
(x * w + b)^2
```

This is a sanity check only. The finite-difference test already verifies the custom implementation without requiring PyTorch.

### 8. Common pitfalls and misconceptions

The notebook demonstrates several failure modes rather than only describing them:

- **Not zeroing gradients:** old gradients accumulate across training iterations and make updates unstable.
- **Overwriting with `=` instead of accumulating with `+=`:** repeated computation paths lose all but the last gradient contribution. The example `b = a + a` should produce `db/da = 2`.
- **Choosing a poor learning rate:** a rate that is too small makes training crawl; a rate that is too large causes oscillation or divergence.
- **Saturated `tanh` activations:** large inputs make the local derivative almost zero, causing vanishing gradients.
- **Confusing backpropagation with optimisation:** backpropagation computes gradients; gradient descent, Adam, RMSProp, and similar methods use those gradients to update parameters.
- **Trusting training loss alone:** a low training loss does not demonstrate generalisation, especially when the model has more parameters than training examples.

The learning-rate example derives the stability condition for the toy quadratic loss. It also shows why the loss curve and held-out performance are more informative than a single training metric.

## Exercise: Add `exp` and Build `sigmoid`

Section 6 contains the student exercise. The initial `exp` cell intentionally raises `NotImplementedError`. Complete the `Value.exp` method by implementing:

1. The forward value using `math.exp(self.data)`.
2. A `_backward()` function using the fact that the derivative of `e^x` is `e^x`.
3. Gradient accumulation through `self.grad += ...`.

The following check compares the result with a centred finite-difference estimate. The notebook then provides a solution cell and constructs:

```text
sigmoid(v) = 1 / (1 + exp(-v))
```

The sigmoid derivative should emerge automatically as `sigmoid(v) * (1 - sigmoid(v))`, even though that derivative is not manually coded into the engine.

When working through the exercise, run the incomplete cell first, implement `exp`, and run the check cell again. The supplied solution is useful for comparison after attempting the task yourself.

## Expected Learning Outcome

The central result of the notebook is that a neural network learns through a repeated, explicit process:

```text
forward pass -> loss -> zero gradients -> backward pass -> parameter update
```

Backpropagation supplies the gradients efficiently by reusing local derivatives throughout the graph. Gradient descent then changes the parameters using those gradients. The final decision boundary is not hand-coded; it emerges from repeatedly applying this process to the training data.

## References

The notebook cites the following sources for the automatic-differentiation design, backpropagation, optimisation, and dataset:

1. Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*.
2. Andrej Karpathy, *micrograd*, GitHub.
3. Rumelhart, Hinton, and Williams, “Learning representations by back-propagating errors,” *Nature* (1986).
4. Goodfellow, Bengio, and Courville, *Deep Learning*, Chapters 6.5 and 8.
5. Baydin et al., “Automatic differentiation in machine learning: a survey.”
6. Michael Nielsen, *Neural Networks and Deep Learning*, Chapter 2.
7. Stanford CS231n optimisation notes.
8. Pedregosa et al., “Scikit-learn: Machine Learning in Python.”