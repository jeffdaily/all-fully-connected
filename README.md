# All Layers, Fully-Connected

This project is derived from [ECP-CANDLE/Benchmarks (P1B1)](https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/P1B1) to experiment with workloads on networks where all layers are fully-connected. The workload is expected to be both compute and transfer intensive. 

### Experiments, P1B3
[Code for N=1 GPU](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B3/p1b3_baseline_keras2.py)

#### Train/Val Loss, 200 Epochs
![TensorBoard Plot](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/results/Capture100-epochs.PNG "Single GPU 200 Epochs")

### Experiments, P1B1
[Code for N=1 GPU](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/p1b1_baseline_keras2.py)

[Code for N=4 GPUs, Data Parallel training](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/4-gpu-p1b1_baseline_keras2.py)

#### Train/Val Loss, 100 Epochs
![TensorBoard Plot](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/results/Capture100-epochs.PNG "Single GPU 100 Epochs")

#### Train/Val Loss, N=4 GPUs, 100 Epochs
![TensorBoard Plot](https://github.com/aurotripathy/all-fully-connected/blob/master/Pilot1/P1B1/results/Capture100-epochs-4-gpu.PNG "N=4 GPUs, 100 Epochs")






