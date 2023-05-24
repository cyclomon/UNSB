# UNSB

Official PyTorch implementation of [Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](www.google.com).

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main2.jpg" />
</p>

Due to the curse of dimensionality, observed samples / training data in high dimensions become sparse and fail to describe image manifolds accurately. Vanilla Schrödinger Bridge learns optimal transport between observed samples, leading to undesirable mappings. We propose **Unpaired Neural Schrödinger Bridge (UNSB)**, which employs adversarial learning and regularization to learn an optimal transport mapping which successfully generalizes beyond observed data. UNSB can be interpreted as successively refining the predicted target domain image, enabling the model to modify fine details while preserving semantics. Here, NFE stands for the number of neural net function evaluations.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main_result_2.jpg" />
</p>

**Code will be released soon.**
