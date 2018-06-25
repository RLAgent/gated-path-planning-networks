# Gated Path Planning Networks (ICML 2018)

This is the official codebase for the following paper:

Lisa Lee\*, Emilio Parisotto\*, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov. **Gated Path Planning Networks**. ICML 2018. https://arxiv.org/abs/1806.06408

## Getting Started

You can clone this repo by running:
```
git clone https://github.com/lileee/gated-path-planning-networks.git
cd gated-path-planning-networks/
```

All subsequent commands in this README should be run from the top-level directory of this repository (i.e., `/path/to/gated-path-planning-networks/`).

### I. Docker container

We provide two Docker containers, with and without GPU support. These containers have Python 3.6.5, PyTorch 0.4.0, and other dependencies installed. They do not contain this codebase or the maze datasets used in our experiments.

To load the container with GPU support:
```
# PyTorch with GPU support
nvidia-docker pull lileee/ubuntu-16.04-pytorch-0.4.0-gpu:v1
nvidia-docker run -v $(pwd):/home --rm -ti lileee/ubuntu-16.04-pytorch-0.4.0-gpu:v1
```

To load the container without GPU support:
```
# PyTorch (CPU-only)
nvidia-docker pull lileee/ubuntu-16.04-pytorch-0.4.0-cpu:v1
nvidia-docker run -v $(pwd):/home --rm -ti lileee/ubuntu-16.04-pytorch-0.4.0-cpu:v1
```

Here is a speed comparison between the Docker containers for training VIN on a `9x9` maze with 5k/1k/1k train-val-test split:

| PyTorch 0.4.0     | time per epoch
|:-----------------:|:---------:|
|with GPU support   | 8.5 sec
|without GPU support| 32.3 sec

### II. Generate a 2D maze dataset

Generate a dataset by running:

```
python generate_dataset.py --output-path mazes.npz --mechanism news --maze-size 9 --train-size 5000 --valid-size 1000 --test-size 1000
```
This will create a datafile `mazes.npz` containing a dataset of `9x9` mazes using the NEWS maze transition mechanism with 5k/1k/1k train-val-test split.

Note:
* The same maze transition mechanism that was used to generate the dataset must be used for `train.py` and `eval.py`. Here, we used  `--mechanism news` to generate the dataset. Other options are `--mechanism moore` and `--mechanism diffdrive`.

### III. Train a model

You can train a VIN with iteration count K=15 and kernel size F=5 on the datafile `mazes.npz` by running:
```
python train.py --datafile mazes.npz --mechanism news --model models.VIN --k 15 --f 5 --save-directory log/vin-k15-f5
```
This will save outputs to the subdirectory `vin-k15-f5/`, including the trained models and learning plots.

Similarly, you can train a GPPN by running:
```
python train.py --datafile mazes.npz --mechanism news --model models.GPPN --k 15 --f 5 --save-directory log/gppn-k15-f5
```

Notes:
* `--mechanism` must be the same as the one used to generate `mazes.npz` (which is `news` in this example).
* `--f` must be an odd integer.

### IV. Evaluate a trained model

Once you have a trained VIN model, you can evaluate it on a dataset by running:

```
python eval.py --datafile mazes.npz --mechanism news --model models.VIN --k 15 --f 5 --load-file log/vin-k15-f5/planner.final.pth
```
Similarly for GPPN:

```
python eval.py --datafile mazes.npz --mechanism news --model models.GPPN --k 15 --f 5 --load-file log/gppn-k15-f5/planner.final.pth
```

Notes:
* `--mechanism` must be the same as the one used to generate `mazes.npz` (which is `news` in this example).
* `--f` must be the same as the one used to train the model.


## Replicating experiments from our paper

### I. Download 2D maze datasets
To replicate experiments from our ICML 2018 paper, first download the datasets by running:
```
./download_datasets.sh
```
This will create a subdirectory `mazes/` containing the following 2D maze datasets used in our experiments:

|         datafile        | maze size | mechanism   | train size | val size | test size |
| :---------------------: |:---------:| :----------:|:----------:|:--------:|:---------:|
| `m15_news_10k.npz`      | `15x15`   | `news`      | 10000      | 2000     | 2000
| `m15_news_25k.npz`      | `15x15`   | `news`      | 25000      | 5000     | 5000
| `m15_news_100k.npz`     | `15x15`   | `news`      | 100000      | 10000    | 10000
| `m15_moore_10k.npz`     | `15x15`   | `moore`     | 10000      | 2000     | 2000
| `m15_moore_25k.npz`     | `15x15`   | `moore`     | 25000      | 5000     | 5000
| `m15_moore_100k.npz`    | `15x15`   | `moore`     | 100000      | 10000    | 10000
| `m15_diffdrive_10k.npz` | `15x15`   | `diffdrive` | 10000      | 2000     | 2000
| `m15_diffdrive_25k.npz` | `15x15`   | `diffdrive` | 25000      | 5000     | 5000
| `m15_diffdrive_100k.npz`| `15x15`   | `diffdrive` | 100000      | 10000    | 10000
| `m28_news_25k.npz`      | `28x28`   | `news`      | 25000      | 5000     | 5000
| `m28_moore_25k.npz`     | `28x28`   | `moore`     | 25000      | 5000     | 5000
| `m28_diffdrive_25k.npz` | `28x28`   | `diffdrive` | 25000      | 5000     | 5000


### II. Train VIN and GPPN
Then you can train VIN with the best (K, F) settings for each dataset from our paper by running:
```
python train.py --datafile mazes/m15_news_10k.npz --mechanism news --model models.VIN --k 30 --f 5
python train.py --datafile mazes/m15_news_25k.npz --mechanism news --model models.VIN --k 20 --f 5
python train.py --datafile mazes/m15_news_100k.npz --mechanism news --model models.VIN --k 30 --f 3

python train.py --datafile mazes/m15_moore_10k.npz --mechanism moore --model models.VIN --k 30 --f 11
python train.py --datafile mazes/m15_moore_25k.npz --mechanism moore --model models.VIN --k 30 --f 5
python train.py --datafile mazes/m15_moore_100k.npz --mechanism moore --model models.VIN --k 30 --f 5

python train.py --datafile mazes/m15_diffdrive_10k.npz --mechanism diffdrive --model models.VIN --k 30 --f 3
python train.py --datafile mazes/m15_diffdrive_25k.npz --mechanism diffdrive --model models.VIN --k 30 --f 3
python train.py --datafile mazes/m15_diffdrive_100k.npz --mechanism diffdrive --model models.VIN --k 30 --f 3

python train.py --datafile mazes/m28_news_25k.npz --mechanism news --model models.VIN --k 56 --f 3
python train.py --datafile mazes/m28_moore_25k.npz --mechanism moore --model models.VIN --k 56 --f 5
python train.py --datafile mazes/m28_diffdrive_25k.npz --mechanism diffdrive --model models.VIN --k 56 --f 3
```

Similarly, you can train GPPN with the best (K, F) settings for each dataset from our paper by running:
```
python train.py --datafile mazes/m15_news_10k.npz --mechanism news --model models.GPPN --k 20 --f 9
python train.py --datafile mazes/m15_news_25k.npz --mechanism news --model models.GPPN --k 20 --f 11
python train.py --datafile mazes/m15_news_100k.npz --mechanism news --model models.GPPN --k 30 --f 11

python train.py --datafile mazes/m15_moore_10k.npz --mechanism moore --model models.GPPN --k 30 --f 7
python train.py --datafile mazes/m15_moore_25k.npz --mechanism moore --model models.GPPN --k 30 --f 9
python train.py --datafile mazes/m15_moore_100k.npz --mechanism moore --model models.GPPN --k 30 --f 7

python train.py --datafile mazes/m15_diffdrive_10k.npz --mechanism diffdrive --model models.GPPN --k 30 --f 11
python train.py --datafile mazes/m15_diffdrive_25k.npz --mechanism diffdrive --model models.GPPN --k 30 --f 9
python train.py --datafile mazes/m15_diffdrive_100k.npz --mechanism diffdrive --model models.GPPN --k 30 --f 9

python train.py --datafile mazes/m28_news_25k.npz --mechanism news --model models.GPPN --k 56 --f 11
python train.py --datafile mazes/m28_moore_25k.npz --mechanism moore --model models.GPPN --k 56 --f 9
python train.py --datafile mazes/m28_diffdrive_25k.npz --mechanism diffdrive --model models.GPPN --k 56 --f 11
```

### III. Test Performance Results

Here are the test performance results from running the above commands inside the Docker container `lileee/ubuntu-16.04-pytorch-0.4.0-gpu:v1`:

<table>
  <tr>
    <td></td>
    <td colspan="4" align="center"><span style="font-weight:bold">VIN</span></td>
    <td colspan="4" align="center"><span style="font-weight:bold">GPPN</span></td>
  </tr>
  <tr>
    <td bgcolor="white" align="center"><span style="font-weight:bold">datafile</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">K</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">F</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">%Opt</span></td>
    <td bgcolor="white"  align="center"><span style="font-weight:bold">%Suc</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">K</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">F</span></td>
    <td bgcolor="white" align="center"><span style="font-weight:bold">%Opt</span></td>
    <td bgcolor="white"  align="center"><span style="font-weight:bold">%Suc</span></td>
  </tr>
<tr>
  <td align="center"><code>m15_news_10k.npz</code></td>
  <td align="center">30</td><td align="center">5</td><td align="center">77.4</td><td align="center">79.0</td>
  <td align="center">20</td><td align="center">9</td><td align="center">96.8</td><td align="center">97.8</td>
</tr>
<tr>
  <td align="center"><code>m15_news_25k.npz</code></td>
  <td align="center">20</td><td align="center">5</td><td align="center">83.6</td><td align="center">84.2</td>
  <td align="center">20</td><td align="center">11</td><td align="center">99.0</td><td align="center">99.3</td>
</tr>
<tr>
  <td align="center"><code>m15_news_100k.npz</code></td>
  <td align="center">30</td><td align="center">3</td><td align="center">92.6</td><td align="center">92.8</td>
  <td align="center">30</td><td align="center">11</td><td align="center">99.7</td><td align="center">99.8</td>
</tr>
<tr>
  <td align="center"><code>m15_moore_10k.npz</code></td>
  <td align="center">30</td><td align="center">11</td><td align="center">86.0</td><td align="center">89.3</td>
  <td align="center">30</td><td align="center">7</td><td align="center">97.0</td><td align="center">98.0</td>
</tr>
<tr>
  <td align="center"><code>m15_moore_25k.npz</code></td>
  <td align="center">30</td><td align="center">5</td><td align="center">85.4</td><td align="center">88.1</td>
  <td align="center">30</td><td align="center">9</td><td align="center">98.9</td><td align="center">99.5</td>
</tr>
<tr>
  <td align="center"><code>m15_moore_100k.npz</code></td>
  <td align="center">30</td><td align="center">5</td><td align="center">96.9</td><td align="center">97.5</td>
  <td align="center">30</td><td align="center">7</td><td align="center">99.6</td><td align="center">99.8</td>
</tr>
<tr>
  <td align="center"><code>m15_diffdrive_10k.npz</code></td>
  <td align="center">30</td><td align="center">3</td><td align="center">98.4</td><td align="center">99.0</td>
  <td align="center">30</td><td align="center">11</td><td align="center">99.1</td><td align="center">99.7</td>
</tr>
<tr>
  <td align="center"><code>m15_diffdrive_25k.npz</code></td>
  <td align="center">30</td><td align="center">3</td><td align="center">96.1</td><td align="center">98.5</td>
  <td align="center">30</td><td align="center">9</td><td align="center">98.9</td><td align="center">99.5</td>
</tr>
<tr>
  <td align="center"><code>m15_diffdrive_100k.npz</code></td>
  <td align="center">30</td><td align="center">3</td><td align="center">99.0</td><td align="center">99.4</td>
  <td align="center">30</td><td align="center">9</td><td align="center">99.8</td><td align="center">99.9</td>
</tr>
<tr>
  <td align="center"><code>m28_news_25k.npz</code></td>
  <td align="center">56</td><td align="center">3</td><td align="center">83.4</td><td align="center">84.2</td>
  <td align="center">56</td><td align="center">11</td><td align="center">96.5</td><td align="center">97.8</td>
</tr>
<tr>
  <td align="center"><code>m28_moore_25k.npz</code></td>
  <td align="center">56</td><td align="center">5</td><td align="center">73.3</td><td align="center">81.0</td>
  <td align="center">56</td><td align="center">9</td><td align="center">96.5</td><td align="center">97.9</td>
</tr>
<tr>
  <td align="center"><code>m28_diffdrive_25k.npz</code></td>
  <td align="center">56</td><td align="center">3</td><td align="center">82.0</td><td align="center">93.6</td>
  <td align="center">56</td><td align="center">11</td><td align="center">95.3</td><td align="center">98.0</td>
</tr>
</table>

Feel free to play around with different iteration counts `--k` and kernel sizes `--f`.

### IV. Version differences (Python, PyTorch)

The test performance results above are slightly different from what is reported in our ICML 2018 paper due to version differences in Python (3.6.5 vs. 2.7.12) and PyTorch (0.4.0 vs. 0.3.1).

Below, we provide instructions to exactly replicate the numbers reported in our ICML 2018 paper.

1. Checkout the Git branch `icml2018`:
```
git checkout icml2018
```

2. Load the Docker container used in our experiments by running:
```
# PyTorch with GPU support
nvidia-docker pull lileee/python-2.7-pytorch-0.3.1-custom:latest
nvidia-docker run -v $(pwd):/home --rm -ti lileee/python-2.7-pytorch-0.3.1-custom:latest
```
This Docker container uses Python 2.7.12 and a custom version of PyTorch 0.3.1 compiled from source at https://github.com/eparisotto/pytorch.

3. Train a model:
```
python train.py --datafile mazes/m15_news_25k.npz --mechanism news --model models.VIN --k 20 --f 5
```

## Citation

If you found this code useful in your research, please cite:

```
@inproceedings{gppn2018,
  author    = {Lisa Lee and Emilio Parisotto and Devendra Singh Chaplot and Eric Xing and Ruslan Salakhutdinov},
  title     = {Gated Path Planning Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML 2018)},
  year      = {2018}
}
```

## Acknowledgments

Thanks to [@kentsommer](https://github.com/kentsommer) for releasing a [PyTorch implementation](https://github.com/kentsommer/pytorch-value-iteration-networks) of the original VIN results, which served as a starting point for this codebase.
