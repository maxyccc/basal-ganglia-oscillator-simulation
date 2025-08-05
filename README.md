# Basal Ganglia Oscillator

A real-time neural network simulation featuring a bipartite spiking neural network with spike-timing dependent plasticity (STDP), designed to model oscillatory dynamics in basal ganglia circuits.

![Simulation Screenshot](snapshot.png)

## Overview

This project implements a computational model of basal ganglia oscillations using a bipartite neural network architecture. The simulation features:

- **Bipartite Network Architecture**: Two interconnected layers with inhibitory (Layer A) and excitatory (Layer B) neurons
- **Leaky Integrate-and-Fire (LIF) Neurons**: Biologically realistic neuron models with membrane dynamics
- **Spike-Timing Dependent Plasticity (STDP)**: Activity-dependent synaptic weight modification
- **Real-time Visualization**: Interactive GUI with raster plots, weight matrices, and population firing rates
- **GPU Acceleration**: Optional CUDA support via CuPy for high-performance computation
- **Interactive Controls**: Real-time parameter adjustment during simulation

## Scientific Context

The basal ganglia are a group of subcortical nuclei crucial for motor control, learning, and decision-making. Oscillatory activity in these circuits, particularly in the beta frequency range (13-30 Hz), has been implicated in both normal function and pathological conditions like Parkinson's disease. This simulation provides a simplified but biologically-inspired model to study:

- Emergence of oscillatory patterns in inhibitory-excitatory networks
- Effects of synaptic plasticity on network dynamics
- Parameter sensitivity in neural oscillations
- Computational mechanisms underlying basal ganglia function

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Dependencies

Install the required packages using pip:

```bash
pip install pygame pygame-gui numpy matplotlib
```

### Optional Dependencies

For GPU acceleration (recommended for larger networks):

```bash
pip install cupy
```

**Note**: CuPy requires CUDA-compatible hardware and CUDA toolkit installation. If CuPy is not available, the simulation will automatically fall back to CPU computation using NumPy.

## Usage

### Basic Usage

Run the main simulation:

```bash
python main.py
```

This will launch the interactive visualization window with default parameters (50 neurons per layer, 30% connectivity).

### GPU/CPU Toggle Demo

To see the runtime switching between CPU and GPU computation:

```bash
python demo_runtime_toggle.py
```

### Interactive Controls

The simulation provides real-time controls for:

- **Connection Probability**: Adjust network sparsity (0.0 - 1.0)
- **Background Rates**: Control external input to each layer (0-50 Hz)
- **STDP Parameters**: Modify learning rates and time constants
- **Weight Parameters**: Adjust synaptic strength distributions
- **Simulation Control**: Pause/resume, reset network, regenerate connections

### Key Features

- **Raster Plots**: Real-time spike visualization for both neuron populations
- **Weight Matrices**: Heatmaps showing synaptic connectivity strength
- **Population Rates**: Time series of firing rates for each layer
- **Network Topology**: Visual representation of connections between layers
- **Performance Monitoring**: FPS and computation mode indicators

## Project Structure

```
BasalGangliaOscillator/
├── main.py                 # Main entry point and simulation launcher
├── network.py              # BipartiteNetwork class with LIF neurons and STDP
├── visualization.py        # NetworkVisualizer class with GUI and plotting
├── demo_runtime_toggle.py  # Demonstration of CPU/GPU switching
├── theme.json              # UI theme configuration
├── snapshot.png            # Example simulation screenshot
└── README.md               # This file
```

### Core Modules

- **`main.py`**: Entry point that initializes and runs the simulation
- **`network.py`**: Contains the `BipartiteNetwork` class implementing:
  - LIF neuron dynamics
  - STDP learning rules
  - CPU/GPU computation switching
  - Vectorized operations for performance
- **`visualization.py`**: Contains the `NetworkVisualizer` class providing:
  - Pygame-based real-time visualization
  - Interactive parameter controls
  - Multiple plot types (raster, heatmaps, time series)
  - Event handling and UI management

## Network Architecture

### Neuron Model

The simulation uses Leaky Integrate-and-Fire (LIF) neurons with:
- Membrane time constant: 20 ms
- Refractory period: 2 ms
- Threshold: -50 mV
- Reset potential: -70 mV
- Resting potential: -65 mV

### Connectivity

- **Layer A → Layer B**: Inhibitory connections (negative weights)
- **Layer B → Layer A**: Excitatory connections (positive weights)
- **Connection Probability**: Adjustable (default: 30%)
- **Weight Distribution**: Log-normal with configurable parameters

### STDP Learning

Spike-timing dependent plasticity modifies synaptic weights based on:
- Pre- and post-synaptic spike timing
- Exponential decay functions
- Separate potentiation and depression time constants
- Configurable learning rates

## Performance

The simulation is optimized for real-time performance:
- **CPU Mode**: Uses NumPy for vectorized operations
- **GPU Mode**: Uses CuPy for CUDA acceleration (when available)
- **Runtime Switching**: Seamless transition between CPU and GPU
- **Typical Performance**: 60 FPS with 100 neurons (50 per layer)

## Requirements

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4 GB RAM minimum, 8 GB recommended
- **Graphics**: OpenGL-compatible graphics card
- **CUDA** (optional): For GPU acceleration

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pygame | ≥2.0.0 | Graphics and GUI framework |
| pygame-gui | ≥0.6.0 | UI elements and controls |
| numpy | ≥1.19.0 | Numerical computations |
| matplotlib | ≥3.3.0 | Colormaps and visualization |
| cupy | ≥8.0.0 | GPU acceleration (optional) |

## Contributing

Contributions are welcome! Areas for potential improvement:

- Additional neuron models (Hodgkin-Huxley, adaptive exponential, etc.)
- More sophisticated plasticity rules (homeostatic, metaplasticity)
- Network topology variations (small-world, scale-free)
- Analysis tools (spectral analysis, connectivity metrics)
- Performance optimizations
- Documentation and examples

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt` (if available)
4. Make your changes
5. Test with both CPU and GPU modes
6. Submit a pull request

## License

This project is distributed under the MIT License. See the LICENSE file for details.

## References

- Leblois, A., Boraud, T., Meissner, W., Bergman, H., & Hansel, D. (2006). Competition between feedback loops underlies normal and pathological dynamics in the basal ganglia. *Journal of Neuroscience*, 26(13), 3567-3583.
- Humphries, M. D., Stewart, R. D., & Gurney, K. N. (2006). A physiologically plausible model of action selection and oscillatory activity in the basal ganglia. *Journal of Neuroscience*, 26(50), 12921-12942.
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the project repository.

---

*This simulation is intended for research and educational purposes. It provides a simplified model of basal ganglia dynamics and should not be used for clinical applications.*
