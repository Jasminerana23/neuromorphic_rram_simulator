# Neuromorphic RRAM Simulator



A Python simulator for RRAM-based neuromorphic computing. It maps trained neural network weights onto a differential RRAM crossbar model and runs inference with a matrix-vector product that follows Ohm's and Kirchhoff's laws. It also includes an adaptive precision controller that selects 64, 128, or 256 conductance levels per input based on prediction confidence.



## Status



- Numpy backpropagation training (64-128-64-10 MLP on sklearn digits; 784-128-64-10 MLP on MNIST)

- Fixed-precision post-training quantization (64, 128, 256 levels)

- Adaptive precision controller with validation-tuned thresholds

- Crossbar pipeline using `CrossbarArray` devices (ideal device)

- Read-noise sweep (multiplicative, sigma 0 to 10%)

- Not yet modeled: device variation, drift, sneak paths, wire resistance, energy/power



## Installation



```bash

git clone https://github.com/Jasminerana23/neuromorphic_rram_simulator.git

cd neuromorphic_rram_simulator

pip install numpy scikit-learn

```



Python 3.8+ tested on Windows.



## Running experiments



Each script runs one experiment and prints results.



```bash

python train_backprop_numpy.py    # numpy training, sklearn digits

python quantize_baseline.py       # fixed 64/128/256 levels

python adaptive_controller.py     # adaptive controller

python multiseed.py               # 5 seeds, mean Â± std

python crossbar_pipeline.py       # weights mapped onto CrossbarArray devices

python crossbar_noise.py          # read-noise sweep

python mnist_numpy.py             # MNIST 784-input (downloads data on first run)

```



## Using your own weights



```python

import numpy as np

from crossbar_array import CrossbarArray



G_MIN, G_MAX = 1/100e3, 1/1e3   # 100 kOhm to 1 kOhm

LOAD = 1000.0



def map_weights(W, L=64):

&#x20;   Wmax = np.abs(W).max()

&#x20;   kp = np.round(np.maximum(W, 0) / Wmax * (L - 1))

&#x20;   kn = np.round(np.maximum(-W, 0) / Wmax * (L - 1))

&#x20;   Gp = G_MIN + (G_MAX - G_MIN) * kp / (L - 1)

&#x20;   Gn = G_MIN + (G_MAX - G_MIN) * kn / (L - 1)

&#x20;   return Gp, Gn, Wmax



def crossbar_matmul(a, Gp, Gn, Wmax):

&#x20;   rows, cols = Gp.shape

&#x20;   cp = CrossbarArray(rows=rows, cols=cols)

&#x20;   cn = CrossbarArray(rows=rows, cols=cols)

&#x20;   cp.set_weights(1.0 / Gp)

&#x20;   cn.set_weights(1.0 / Gn)

&#x20;   Gp_read = 1.0 / cp.get_weights()

&#x20;   Gn_read = 1.0 / cn.get_weights()

&#x20;   return (a @ (Gp_read - Gn_read)) * Wmax / (G_MAX - G_MIN)

```



## Limitations



- Training runs in numpy, not on the crossbar. Weights are mapped after training.

- `RRAMDevice.write()` switching model is not calibrated; weights are loaded with `set_weights()`.

- "Average levels" is a proxy for precision, not a measured energy or power result.

- Results are on an ideal device, plus multiplicative read noise only.

- `CrossbarArray` creates one object per device and is slow for large matrices; the pipeline uses vectorized math.



## Files



- `rram_device.py`: single RRAM device (resistance range, read, write)

- `crossbar_array.py`: crossbar of RRAM devices, forward pass, weight load

- `backpropagation.py`, `neuron_model.py`, `learning_rules.py`, `train_network.py`: earlier Week 2-3 components

- `train_backprop_numpy.py`, `quantize_baseline.py`, `adaptive_controller.py`, `multiseed.py`, `crossbar_pipeline.py`, `crossbar_noise.py`, `mnist_numpy.py`: experiment scripts



## License



MIT (see LICENSE)


