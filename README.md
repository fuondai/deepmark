# Beyond Robustness: On the Unforgeability Trade-offs in Statistical DNN Watermarking

This repository contains the source code for the paper *Beyond Robustness: On the Unforgeability Trade-offs in Statistical DNN Watermarking*, presented at VCRIS 2025.

## Abstract

Statistical watermarking frameworks for Deep Neural Networks (DNNs) often rely on a regularization term to embed a verifiable signature. While this technique is known to improve robustness against removal attacks, its impact on unforgeability remains poorly understood. This paper provides the first empirical demonstration that the very regularization mechanism employed to enhance watermark robustness paradoxically creates the statistical conditions for successful forgery. Focusing on the DeepMark framework, we show that strong regularization, while decreasing the p-value for the owner's key, can inadvertently increase the likelihood of a random key also yielding a statistically significant result. Our analysis reveals this is not a flaw in the statistical test itself, but a manifestation of spurious correlations in the high-dimensional weight space, a vulnerability exacerbated by the embedding process. We conclude that statistical significance alone is a fragile and insufficient proxy for security in these systems. Our work serves as a critical analysis and cautionary tale, urging the community to look beyond surface-level metrics and integrate cryptographic principles for reliable ownership verification.

## Code Structure

*   `deepmark.py`: This file contains the core implementation of the DeepMark statistical watermarking framework as described in the paper. It includes the logic for watermark embedding and verification.
*   `attacks.py`: This file implements the forgery attack described in the paper. It contains the necessary functions to test the unforgeability of a watermarked model.
*   `run_experiments.py`: This script executes the experiments presented in the paper. It trains a model, embeds a watermark using `deepmark.py`, and then runs the forgery attack from `attacks.py` to reproduce the "ownership deadlock" phenomenon.

## Running the Experiments

To replicate the findings of the paper, you can run the main experiment script:

```bash
python run_experiments.py
```

This will perform the full workflow:

1.  Train a DNN on the specified dataset.
2.  Embed a watermark into the model's weights using the DeepMark methodology.
3.  Perform a verification check to confirm the owner's key.
4.  Launch a forgery attack by testing a large number of random keys against the watermarked model.
5.  Print the p-values for both the owner's key and any successful forgery attempts, demonstrating the trade-offs discussed in the paper.

For more detailed configuration, please refer to the arguments within `run_experiments.py`.
