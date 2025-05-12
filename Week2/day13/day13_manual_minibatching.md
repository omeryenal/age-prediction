# Day 13 – Manual Mini-Batching

## Goals for Today
- Understand the role of mini-batches in model training
- Learn how batching improves gradient descent
- Implement mini-batch logic from scratch using NumPy
- Tune batch size and observe its effect
- Prepare for batch-wise training in deep learning

---

## Lecture 1 – Why Use Mini-Batches?

Gradient Descent Variants:
- **Batch GD:** Uses all data for each update → accurate but slow
- **Stochastic GD (SGD):** Uses one sample per update → fast but noisy
- **Mini-Batch GD:** Uses small batches (e.g., 32, 64) → balance between speed and stability

**Benefits of mini-batches:**
- Faster convergence
- Memory efficiency
- Smooth updates (less noisy than SGD, faster than batch GD)

---

## Lecture 2 – Choosing the Right Batch Size

There's no universal rule, but common sizes:
- 16, 32, 64, 128, 256

**Trade-offs:**
- Smaller batch → noisy updates, more regularization
- Larger batch → smoother updates, may overfit

Start with **32 or 64** and tune later.

**Note:** Batch size affects:
- Convergence speed
- Model generalization
- GPU usage

---

## Lecture 3 – Building a Mini-Batch Generator (NumPy)

```python
def generate_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
for X_batch, y_batch in generate_batches(X_train, y_train, batch_size=32):
    # Do gradient step here
    ...
```
## Lecture 4 – Epochs vs Iterations vs Steps

Epoch: One full pass through the training dataset
Iteration: One update step (one mini-batch)
Steps per epoch: num_samples / batch_size
In training logs you often see:

Epoch 5/10, Step 8/100, Loss: ...
Tracking both epochs and steps helps diagnose training problems.

## Lecture 5 – Mini-Batching in Real Training Loops
```python
W, b = initialize_weights()
epochs = 10
batch_size = 32

for epoch in range(epochs):
    for X_batch, y_batch in generate_batches(X_train, y_train, batch_size):
        y_pred = model(X_batch, W, b)
        loss = compute_loss(y_pred, y_batch)
        dW, db = compute_gradients(X_batch, y_batch, y_pred)
        W, b = update_weights(W, b, dW, db)
```
Mini-batching works seamlessly with manual models and is essential for scalable training.

Reflection

Today I implemented my own batching system and better understood how deep learning frameworks train models in pieces. Mini-batches give me the best of both worlds: stability and speed. I now feel confident building full training loops.

