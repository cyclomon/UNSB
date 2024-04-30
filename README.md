# Unpaired Neural Schrödinger Bridge (ICLR 2024)

Official PyTorch implementation of [Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](https://arxiv.org/abs/2305.15086) by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, [Gihyun Kwon](https://scholar.google.co.kr/citations?user=yexbg8gAAAAJ&hl=en)\*, [Kwanyoung Kim](https://sites.google.com/view/kwanyoung-kim/), and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en) (\*Equal contribution), **accepted to ICLR 2024**.


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

## Environment
```
$ conda create -n UNSB python=3.8
$ conda activate UNSB
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ conda install -c conda-forge packaging 
$ conda install -c "conda-forge/label/cf201901" visdom 
$ conda install -c conda-forge gputil 
$ conda install -c conda-forge dominate 
```

## Dataset Download
Download the dataset with following script e.g.

```
bash ./datasets/download_cut_dataset.sh horse2zebra
```

Due to copyright issue, we do not directly provide cityscapes dataset. 
please refer to the original repository of [CUT](https://github.com/taesungp/contrastive-unpaired-translation).

## Training 
Refer the ```./run_train.sh``` file or

```
python train.py --dataroot ./datasets/horse2zebra --name h2z_SB \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0
```

for cityscapes and map2sat, 

```
python train.py --dataroot ./datasets/cityscapes --name city_SB \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --direction B2A
```

for summer2winter,

```
cd vgg_sb
bash ./scripts/train_sc_sim2win_main.sh
```

Although the training is available with arbitrary batch size, we recommend to use batch size = 1.

## Test & Evaluation
Refer the ```./run_test.sh``` file or 

```
python test.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase test --epoch [epoch-for-test] --eval --num_test [num-test-image] \
--gpu_ids 0 --checkpoints_dir ./checkpoints
```

The outputs will be saved in ```./results/[experiment-name]/```

Folders named as ```fake_[num_NFE]``` represent the generated outputs with different NFE steps.

For evaluation, we use official module of [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

```
python -m pytorch_fid [output-path] [real-path]
```

```real-path``` should be test images of target domain. 

For testing on our vgg-based trained model, 

Refer the ```./vgg_sb/scripts/test_sc_main.sh``` file 

The pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/1Q8tuBGegMMHd9PzvcklDm0wM1sE4PPwK?usp=sharing)

## References

If you find this paper useful for your research, please consider citing
```bib
@InProceedings{
  kim2023unsb,
  title={Unpaired Image-to-Image Translation via Neural Schrödinger Bridge},
  author={Beomsu Kim and Gihyun Kwon and Kwanyoung Kim and Jong Chul Ye},
  booktitle={ICLR},
  year={2024}
}
```
### Acknowledgement
Our source code is based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation). \
We thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID calculation. \
We modified the network based on the implementation of [DDGAN](https://github.com/NVlabs/denoising-diffusion-gan).
