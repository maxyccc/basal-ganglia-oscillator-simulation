"""
Main Entry Point for Basal Ganglia Oscillator Simulation

This module serves as the main entry point for the Basal Ganglia Oscillator
simulation. It imports and orchestrates the separated modules to create and
run the neural network visualization.

The simulation features:
- Bipartite neural network with LIF neurons
- Spike-timing dependent plasticity (STDP)
- Real-time visualization with raster plots and weight matrices
- Interactive controls for simulation parameters

Usage:
    python main.py

Dependencies:
    - pygame
    - pygame_gui
    - numpy
    - matplotlib
"""

from visualization import NetworkVisualizer


def main():
    """
    Main function to initialize and run the neural network simulation.
    
    Creates a NetworkVisualizer instance and starts the main simulation loop.
    The visualizer handles all aspects of the simulation including:
    - Neural network creation and dynamics
    - Real-time plotting and visualization
    - User interface and event handling
    """
    try:
        # Create and run the network visualizer
        simulator = NetworkVisualizer()
        simulator.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        raise


if __name__ == "__main__":
    main()
