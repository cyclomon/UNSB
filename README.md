# Unpaired Neural Schrödinger Bridge

Official PyTorch implementation of [Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](https://arxiv.org/abs/2305.15086) by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, [Gihyun Kwon](https://scholar.google.co.kr/citations?user=yexbg8gAAAAJ&hl=en)\*, [Kwanyoung Kim](https://sites.google.com/view/kwanyoung-kim/), and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en). (\*Equal contribution)

**Code will be released soon, so stay tuned!**

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/gif.gif" />
</p>

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main2.jpg" />
</p>

Due to the curse of dimensionality, observed samples / training data in high dimensions become sparse and fail to describe image manifolds accurately. Vanilla Schrödinger Bridge learns optimal transport between observed samples, leading to undesirable mappings.

We propose the **Unpaired Neural Schrödinger Bridge (UNSB)**, which employs adversarial learning and regularization to learn an optimal transport mapping which successfully generalizes beyond observed data. UNSB can be interpreted as successively refining the predicted target domain image, enabling the model to modify fine details while preserving semantics. Here, NFE stands for the number of neural net function evaluations.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/main_result_2.jpg" />
</p>

Quantitatively, out method out-performed all one-step baseline methods based on GANs.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/table.png" width="80%" height="80%" />
</p>

The superior performance of UNSB can be attributed to the fact that UNSB generates images in multiple stages. Indeed, we observe in the graph below that sample quality improves with more NFEs.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/NFE_FID.png" width="40%" height="40%" />
</p>

However, occasionally, too much NFEs led to "over-translation", where the target domain style is excessively applied to the source image. A failure case is shown below. This may be the reason behind increasing FID for some datasets at NFEs 4 or 5.

<p align="center">
  <img src="https://github.com/cyclomon/UNSB/blob/main/assets/Main_failure.png" width="40%" height="40%" />
</p>

## References

If you find this paper useful for your research, please consider citing
```bib
@article{
  kim2023unsb,
  title={Unpaired Image-to-Image Translation via Neural Schrödinger Bridge},
  author={Beomsu Kim and Gihyun Kwon and Kwanyoung Kim and Jong Chul Ye},
  journal={arxiv preprint arXiv:2305.15086},
  year={2023}
}
```
