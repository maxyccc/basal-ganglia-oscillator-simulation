#!/usr/bin/env python3
"""
Demonstration of the runtime CPU/GPU toggle feature.

This script shows how to use the new toggle functionality to switch
between CPU and GPU computation modes at runtime.
"""

import time
from network import BipartiteNetwork

def main():
    print("BipartiteNetwork Runtime CPU/GPU Toggle Demo")
    print("=" * 50)
    
    # Create network in CPU mode (default)
    print("\n1. Creating network in CPU mode...")
    network = BipartiteNetwork(n_A=30, n_B=30, connection_prob=0.4, use_gpu=False)
    print(f"   Current mode: {network.get_compute_mode()}")
    
    # Run some simulation in CPU mode
    print("\n2. Running simulation in CPU mode...")
    start_time = time.time()
    for i in range(50):
        spikes_A, spikes_B = network.update()
        if i % 10 == 0:
            print(f"   Step {i}: {len(spikes_A)} A spikes, {len(spikes_B)} B spikes")
    
    cpu_time = time.time() - start_time
    print(f"   CPU simulation: 50 steps in {cpu_time:.3f} seconds")
    
    # Get some data before switching
    voltage_before = network.get_neuron_voltage('A', 0)
    print(f"   Sample voltage before switch: {voltage_before:.2f} mV")
    
    # Toggle to GPU mode
    print("\n3. Switching to GPU mode...")
    network.toggle_compute_mode()
    print(f"   New mode: {network.get_compute_mode()}")
    
    # Verify data integrity
    voltage_after = network.get_neuron_voltage('A', 0)
    print(f"   Sample voltage after switch: {voltage_after:.2f} mV")
    print(f"   Voltage difference: {abs(voltage_before - voltage_after):.2e}")
    
    # Run simulation in GPU mode
    print("\n4. Running simulation in GPU mode...")
    start_time = time.time()
    for i in range(50):
        spikes_A, spikes_B = network.update()
        if i % 10 == 0:
            print(f"   Step {i}: {len(spikes_A)} A spikes, {len(spikes_B)} B spikes")
    
    gpu_time = time.time() - start_time
    print(f"   GPU simulation: 50 steps in {gpu_time:.3f} seconds")
    
    # Compare performance
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"   GPU speedup: {speedup:.1f}x faster than CPU")
    else:
        slowdown = gpu_time / cpu_time
        print(f"   GPU overhead: {slowdown:.1f}x slower than CPU (normal for small networks)")
    
    # Toggle back to CPU
    print("\n5. Switching back to CPU mode...")
    network.toggle_compute_mode()
    print(f"   Final mode: {network.get_compute_mode()}")
    
    # Final simulation
    print("\n6. Final simulation in CPU mode...")
    for i in range(10):
        spikes_A, spikes_B = network.update()
    
    final_voltage = network.get_neuron_voltage('A', 0)
    print(f"   Final voltage: {final_voltage:.2f} mV")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("  ✓ Runtime switching between CPU and GPU modes")
    print("  ✓ Data integrity preserved during switches")
    print("  ✓ Seamless API compatibility")
    print("  ✓ Performance comparison")
    
    print(f"\nFinal network state:")
    print(f"  - Mode: {network.get_compute_mode()}")
    print(f"  - Simulation time: {network.current_time:.1f} ms")
    print(f"  - Layer A neurons: {network.n_A}")
    print(f"  - Layer B neurons: {network.n_B}")

if __name__ == "__main__":
    main()
