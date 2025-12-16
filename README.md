NeuralDbg MVP

Conceptually:

- We wrap the model to capture every forward/backward pass (event trace).

- We version the tensors to inspect changes over time.

- We set one breakpoint to detect vanishing gradients.

- We can query the state to understand why training failed.