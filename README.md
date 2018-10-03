# All Layers Fully-Connected

This project is derived from https://github.com/ECP-CANDLE/Benchmarks to experiment with workloads on networks where all layers are fully-connected.

### Experiments
[Code for N=1 GPU, Data Parallel training](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/p1b1_baseline_keras2.py)

[Code for N=4 GPUs, Data Parallel training](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/4-gpu-p1b1_baseline_keras2.py)

#### Train/Val Loss, 100 Epochs
![TensorBoard Plot](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/results/Capture100-epochs.PNG "Single GPU 100 Epochs")

#### Train/Val Loss, 400 Epochs
![TensorBoard Plot](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/results/Capture400-epochs.PNG "N=4 GPUs 400 Epochs")






