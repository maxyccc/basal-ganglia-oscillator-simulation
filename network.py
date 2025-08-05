"""
Neural Network Module

This module contains the implementation of the bipartite neural network with
spike-timing dependent plasticity (STDP) for the Basal Ganglia Oscillator simulation.

Classes:
    BipartiteNetwork: A two-layer neural network with STDP learning (CPU/GPU switchable implementation)
"""

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback alias
    CUPY_AVAILABLE = False

import numpy as np


class BipartiteNetwork:
    """
    Bipartite neural network with spike-timing dependent plasticity (CPU/GPU switchable implementation).

    This class implements a two-layer neural network where Layer A contains inhibitory
    neurons and Layer B contains excitatory neurons. The network features bidirectional
    connectivity with STDP learning rules that modify synaptic weights based on
    spike timing correlations.

    This implementation supports runtime switching between CPU (NumPy) and GPU (CuPy) computation.
    All neuron dynamics and STDP calculations are performed using vectorized operations on the
    selected backend.

    Attributes:
        n_A (int): Number of neurons in Layer A (inhibitory)
        n_B (int): Number of neurons in Layer B (excitatory)
        connection_prob (float): Probability of connection between any two neurons
        use_gpu (bool): Whether to use GPU acceleration (requires CuPy)
        xp (module): Current computation backend (cp for GPU, np for CPU)
        V_m_A (ndarray): Membrane potentials for Layer A neurons
        V_m_B (ndarray): Membrane potentials for Layer B neurons
        last_spike_time_A (ndarray): Last spike times for Layer A neurons
        last_spike_time_B (ndarray): Last spike times for Layer B neurons
        spike_history_A (ndarray): Spike history for Layer A neurons (for STDP)
        spike_history_B (ndarray): Spike history for Layer B neurons (for STDP)
        W_BA (ndarray): Weight matrix from Layer B to Layer A (excitatory)
        W_AB (ndarray): Weight matrix from Layer A to Layer B (inhibitory)
        current_time (float): Current simulation time in ms
        dt (float): Time step for simulation in ms
        background_rate_A (float): Background Poisson input rate for Layer A in Hz
        background_rate_B (float): Background Poisson input rate for Layer B in Hz
    """
    
    def __init__(self, n_A=50, n_B=50, connection_prob=0.3, use_gpu=False):
        """
        Initialize the bipartite network with specified parameters.

        Args:
            n_A (int, optional): Number of inhibitory neurons in Layer A. Defaults to 50.
            n_B (int, optional): Number of excitatory neurons in Layer B. Defaults to 50.
            connection_prob (float, optional): Connection probability. Defaults to 0.3.
            use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
        """
        self.n_A = n_A  # Layer A (inhibitory)
        self.n_B = n_B  # Layer B (excitatory)
        self.connection_prob = connection_prob

        # Initialize compute backend
        self._set_compute_backend(use_gpu)

        # Lognormal distribution parameters for weight initialization
        # W_BA (excitatory) parameters
        self.lognorm_mu_BA = 0.5  # Mean of underlying normal distribution
        self.lognorm_sigma_BA = 0.5  # Standard deviation of underlying normal distribution
        self.lognorm_scale_BA = 0.25  # Scale factor for resulting lognormal values

        # W_AB (inhibitory) parameters
        self.lognorm_mu_AB = 0.5  # Mean of underlying normal distribution
        self.lognorm_sigma_AB = 0.5  # Standard deviation of underlying normal distribution
        self.lognorm_scale_AB = 0.25  # Scale factor for resulting lognormal values

        # Vectorized neuron parameters (same for all neurons in each layer)
        # LIF parameters from original LIFNeuron class
        self.V_rest = -70.0  # mV
        self.V_threshold = -50.0  # mV
        self.V_reset = -80.0  # mV
        self.tau_m = 10.0  # ms, membrane time constant
        self.R_m = 10.0  # MOhm, membrane resistance
        self.refractory_period = 2.0  # ms

        # Vectorized neuron state arrays (backend-agnostic)
        # Membrane potentials
        self.V_m_A = self.xp.full(n_A, self.V_rest, dtype=self.xp.float64)
        self.V_m_B = self.xp.full(n_B, self.V_rest, dtype=self.xp.float64)

        # Last spike times (initialized to far past)
        self.last_spike_time_A = self.xp.full(n_A, -1000.0, dtype=self.xp.float64)
        self.last_spike_time_B = self.xp.full(n_B, -1000.0, dtype=self.xp.float64)

        # Refractory period tracking (time when refractory period ends)
        self.refractory_until_A = self.xp.full(n_A, -1000.0, dtype=self.xp.float64)
        self.refractory_until_B = self.xp.full(n_B, -1000.0, dtype=self.xp.float64)

        # Spike history for STDP (circular buffer approach)
        self.max_spike_history = 100  # Maximum spikes to remember per neuron
        self.spike_history_A = self.xp.full((n_A, self.max_spike_history), -1000.0, dtype=self.xp.float64)
        self.spike_history_B = self.xp.full((n_B, self.max_spike_history), -1000.0, dtype=self.xp.float64)
        self.spike_history_idx_A = self.xp.zeros(n_A, dtype=self.xp.int32)  # Current index in circular buffer
        self.spike_history_idx_B = self.xp.zeros(n_B, dtype=self.xp.int32)  # Current index in circular buffer

        # For backward compatibility, create empty layer attributes
        # These are no longer used but may be referenced by external code
        self.layer_A = None  # Deprecated: use vectorized arrays instead
        self.layer_B = None  # Deprecated: use vectorized arrays instead

        # STDP parameters
        self.A_LTP = 0.01  # Long-term potentiation amplitude
        self.A_LTD = 0.0105  # Long-term depression amplitude (balanced to prevent runaway excitation)
        self.tau_LTP = 20.0  # LTP time constant in ms
        self.tau_LTD = 20.0  # LTD time constant in ms

        # Synaptic parameters
        self.synaptic_strength = 15.0  # Base synaptic strength

        # Initialize connectivity matrices
        self.initialize_connections()

        # Simulation parameters
        self.dt = 1.0  # Time step in ms
        self.current_time = 0.0
        self.background_rate_A = 5.0  # Background Poisson input rate for Layer A in Hz
        self.background_rate_B = 5.0  # Background Poisson input rate for Layer B in Hz

        # Pre-allocate spike arrays for performance (backend-agnostic)
        self.spikes_A_arr = self.xp.zeros(self.n_A, dtype=bool)
        self.spikes_B_arr = self.xp.zeros(self.n_B, dtype=bool)

    def _set_compute_backend(self, use_gpu):
        """
        Set the computation backend (CPU or GPU).

        Args:
            use_gpu (bool): Whether to use GPU acceleration
        """
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: GPU requested but CuPy not available. Falling back to CPU.")
            use_gpu = False

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.xp = cp
            print("Using GPU acceleration (CuPy)")
        else:
            self.xp = np
            print("Using CPU computation (NumPy)")

    def get_compute_mode(self):
        """
        Get the current computation mode.

        Returns:
            str: "GPU" if using GPU acceleration, "CPU" if using CPU
        """
        return "GPU" if self.use_gpu else "CPU"

    def toggle_compute_mode(self):
        """
        Toggle between CPU and GPU computation modes.

        This method transfers all arrays between CPU and GPU memory as needed.
        """
        new_use_gpu = not self.use_gpu

        if new_use_gpu and not CUPY_AVAILABLE:
            print("Warning: Cannot switch to GPU mode - CuPy not available.")
            return

        print(f"Switching from {self.get_compute_mode()} to {'GPU' if new_use_gpu else 'CPU'} mode...")

        # Transfer all arrays to the new backend
        self._transfer_arrays_to_backend(new_use_gpu)

        # Update backend
        self._set_compute_backend(new_use_gpu)

        print(f"Successfully switched to {self.get_compute_mode()} mode.")

    def _transfer_arrays_to_backend(self, use_gpu):
        """
        Transfer all network arrays to the specified backend.

        Args:
            use_gpu (bool): Whether to transfer to GPU (True) or CPU (False)
        """
        if use_gpu:
            # Transfer to GPU using CuPy
            self.V_m_A = cp.asarray(self.V_m_A)
            self.V_m_B = cp.asarray(self.V_m_B)
            self.last_spike_time_A = cp.asarray(self.last_spike_time_A)
            self.last_spike_time_B = cp.asarray(self.last_spike_time_B)
            self.refractory_until_A = cp.asarray(self.refractory_until_A)
            self.refractory_until_B = cp.asarray(self.refractory_until_B)

            # Transfer spike history arrays
            self.spike_history_A = cp.asarray(self.spike_history_A)
            self.spike_history_B = cp.asarray(self.spike_history_B)
            self.spike_history_idx_A = cp.asarray(self.spike_history_idx_A)
            self.spike_history_idx_B = cp.asarray(self.spike_history_idx_B)

            # Transfer weight matrices
            self.W_BA = cp.asarray(self.W_BA)
            self.W_AB = cp.asarray(self.W_AB)

            # Transfer spike arrays
            self.spikes_A_arr = cp.asarray(self.spikes_A_arr)
            self.spikes_B_arr = cp.asarray(self.spikes_B_arr)
        else:
            # Transfer to CPU using explicit .get() for CuPy arrays
            self.V_m_A = self._to_cpu_array(self.V_m_A)
            self.V_m_B = self._to_cpu_array(self.V_m_B)
            self.last_spike_time_A = self._to_cpu_array(self.last_spike_time_A)
            self.last_spike_time_B = self._to_cpu_array(self.last_spike_time_B)
            self.refractory_until_A = self._to_cpu_array(self.refractory_until_A)
            self.refractory_until_B = self._to_cpu_array(self.refractory_until_B)

            # Transfer spike history arrays
            self.spike_history_A = self._to_cpu_array(self.spike_history_A)
            self.spike_history_B = self._to_cpu_array(self.spike_history_B)
            self.spike_history_idx_A = self._to_cpu_array(self.spike_history_idx_A)
            self.spike_history_idx_B = self._to_cpu_array(self.spike_history_idx_B)

            # Transfer weight matrices
            self.W_BA = self._to_cpu_array(self.W_BA)
            self.W_AB = self._to_cpu_array(self.W_AB)

            # Transfer spike arrays
            self.spikes_A_arr = self._to_cpu_array(self.spikes_A_arr)
            self.spikes_B_arr = self._to_cpu_array(self.spikes_B_arr)

    def _to_cpu_array(self, array):
        """
        Convert array to CPU (NumPy) format.

        Args:
            array: Input array (either NumPy or CuPy)

        Returns:
            np.ndarray: NumPy array
        """
        if hasattr(array, 'get'):
            # It's a CuPy array, transfer to CPU
            return array.get()
        else:
            # It's already a NumPy array
            return array

    def initialize_connections(self):
        """
        Initialize sparse random connectivity between layers using lognormal distribution.

        Creates two weight matrices and persistent connectivity masks:
        - W_BA: Excitatory connections from Layer B to Layer A (using lognormal distribution)
        - W_AB: Inhibitory connections from Layer A to Layer B (using lognormal distribution)
        - connection_mask_BA: Boolean mask indicating which B->A connections exist (persistent)
        - connection_mask_AB: Boolean mask indicating which A->B connections exist (persistent)

        The connectivity masks preserve the original sparse topology and are never modified
        during STDP updates, allowing weights to reach zero while remaining modifiable.
        """
        # W_BA: excitatory connections from B to A using lognormal distribution
        rand_matrix_BA = self.xp.random.rand(self.n_A, self.n_B)
        lognorm_weights_BA = self.xp.random.lognormal(
            mean=self.lognorm_mu_BA,
            sigma=self.lognorm_sigma_BA,
            size=(self.n_A, self.n_B)
        ) * self.lognorm_scale_BA

        # Create persistent connectivity mask for B->A connections
        self.connection_mask_BA = rand_matrix_BA < self.connection_prob

        # Initialize weights only where connections exist
        self.W_BA = self.xp.where(
            self.connection_mask_BA,
            lognorm_weights_BA,
            0
        )

        # W_AB: inhibitory connections from A to B using lognormal distribution
        rand_matrix_AB = self.xp.random.rand(self.n_B, self.n_A)
        lognorm_weights_AB = self.xp.random.lognormal(
            mean=self.lognorm_mu_AB,
            sigma=self.lognorm_sigma_AB,
            size=(self.n_B, self.n_A)
        ) * self.lognorm_scale_AB

        # Create persistent connectivity mask for A->B connections
        self.connection_mask_AB = rand_matrix_AB < self.connection_prob

        # Initialize weights only where connections exist (make inhibitory weights negative)
        self.W_AB = self.xp.where(
            self.connection_mask_AB,
            -lognorm_weights_AB,  # Negative for inhibitory connections
            0
        )

    def update_lognormal_parameters(self, mu_BA=None, sigma_BA=None, scale_BA=None,
                                   mu_AB=None, sigma_AB=None, scale_AB=None):
        """
        Update lognormal distribution parameters and regenerate weight matrices.

        Args:
            mu_BA (float, optional): Mean for W_BA lognormal distribution
            sigma_BA (float, optional): Standard deviation for W_BA lognormal distribution
            scale_BA (float, optional): Scale factor for W_BA lognormal distribution
            mu_AB (float, optional): Mean for W_AB lognormal distribution
            sigma_AB (float, optional): Standard deviation for W_AB lognormal distribution
            scale_AB (float, optional): Scale factor for W_AB lognormal distribution
        """
        # Update W_BA parameters if provided
        if mu_BA is not None:
            self.lognorm_mu_BA = mu_BA
        if sigma_BA is not None:
            self.lognorm_sigma_BA = sigma_BA
        if scale_BA is not None:
            self.lognorm_scale_BA = scale_BA

        # Update W_AB parameters if provided
        if mu_AB is not None:
            self.lognorm_mu_AB = mu_AB
        if sigma_AB is not None:
            self.lognorm_sigma_AB = sigma_AB
        if scale_AB is not None:
            self.lognorm_scale_AB = scale_AB

        # Regenerate weight matrices with new parameters
        self.initialize_connections()

    def reset_network_state(self):
        """
        Reset all neuron states and simulation time to initial conditions.

        This method resets:
        - Membrane potentials to resting potential
        - Last spike times to far past
        - Refractory periods to far past
        - Spike history buffers
        - Simulation time to 0
        - Spike arrays

        This is called when the reset button is pressed to ensure the network
        returns to a clean initial state while preserving connectivity.
        """
        # Reset simulation time
        self.current_time = 0.0

        # Reset membrane potentials to resting potential
        self.V_m_A.fill(self.V_rest)
        self.V_m_B.fill(self.V_rest)

        # Reset last spike times to far past
        self.last_spike_time_A.fill(-1000.0)
        self.last_spike_time_B.fill(-1000.0)

        # Reset refractory periods to far past (no neurons in refractory)
        self.refractory_until_A.fill(-1000.0)
        self.refractory_until_B.fill(-1000.0)

        # Clear spike history buffers
        self.spike_history_A.fill(-1000.0)
        self.spike_history_B.fill(-1000.0)

        # Reset spike history circular buffer indices
        self.spike_history_idx_A.fill(0)
        self.spike_history_idx_B.fill(0)

        # Reset current spike arrays
        self.spikes_A_arr.fill(False)
        self.spikes_B_arr.fill(False)

    def poisson_input(self, rate_hz, dt, n_neurons):
        """
        Generate Poisson spike train for a population of neurons.

        Args:
            rate_hz (float): Firing rate in Hz
            dt (float): Time step in ms
            n_neurons (int): Number of neurons

        Returns:
            ndarray: Boolean array indicating which neurons received input
        """
        prob = rate_hz * (dt / 1000.0)
        return self.xp.random.random(n_neurons) < prob
    
    def _update_spike_history(self, layer_indices, spike_times, layer_name):
        """
        Update spike history for neurons that spiked (GPU-optimized).

        Args:
            layer_indices (cp.ndarray): Indices of neurons that spiked
            spike_times (float): Current time when spikes occurred
            layer_name (str): 'A' or 'B' to specify which layer
        """
        if len(layer_indices) == 0:
            return

        if layer_name == 'A':
            spike_history = self.spike_history_A
            spike_idx = self.spike_history_idx_A
        else:
            spike_history = self.spike_history_B
            spike_idx = self.spike_history_idx_B

        # Vectorized update using advanced indexing
        current_indices = spike_idx[layer_indices]
        spike_history[layer_indices, current_indices] = spike_times

        # Update circular buffer indices (vectorized)
        spike_idx[layer_indices] = (current_indices + 1) % self.max_spike_history

    def _vectorized_stdp_calculation(self, post_spike_indices, pre_layer_name, post_layer_name, weight_matrix, connectivity_mask):
        """
        Fully vectorized STDP calculation for a set of post-synaptic spikes (backend-optimized).

        This implementation eliminates all Python loops and uses backend broadcasting
        and advanced indexing for maximum performance. Uses persistent connectivity masks
        to distinguish between structural zeros (no connection) and learned zeros (weak connections).

        Args:
            post_spike_indices (ndarray): Indices of post-synaptic neurons that spiked
            pre_layer_name (str): 'A' or 'B' for pre-synaptic layer
            post_layer_name (str): 'A' or 'B' for post-synaptic layer
            weight_matrix (ndarray): Weight matrix to update
            connectivity_mask (ndarray): Boolean mask indicating which connections exist (persistent)

        Returns:
            ndarray: Weight changes to apply
        """
        if len(post_spike_indices) == 0:
            return self.xp.zeros_like(weight_matrix)

        # Get spike history for pre-synaptic layer
        if pre_layer_name == 'A':
            pre_spike_history = self.spike_history_A
        else:
            pre_spike_history = self.spike_history_B

        # Initialize delta weights
        delta_weights = self.xp.zeros_like(weight_matrix)

        # Get connectivity mask for post-synaptic neurons that spiked using persistent topology
        # Shape: (len(post_spike_indices), n_pre_neurons)
        post_connectivity_slice = connectivity_mask[post_spike_indices, :]

        # Early exit if no connections exist
        if not self.xp.any(post_connectivity_slice):
            return delta_weights

        # Find all (post_idx_in_slice, pre_idx) pairs with connections
        post_indices_in_slice, pre_indices = self.xp.where(post_connectivity_slice)

        if len(post_indices_in_slice) == 0:
            return delta_weights

        # Map back to original post indices
        actual_post_indices = post_spike_indices[post_indices_in_slice]

        # Get spike histories for all connected pre-synaptic neurons
        # Shape: (n_connections, max_spike_history)
        connected_pre_histories = pre_spike_history[pre_indices, :]

        # Create valid spike mask (spike times > -999.0)
        # Shape: (n_connections, max_spike_history)
        valid_spike_mask = connected_pre_histories > -999.0

        # Early exit if no valid spikes
        if not self.xp.any(valid_spike_mask):
            return delta_weights

        # Calculate time differences for all valid spikes
        # Broadcasting: (n_connections, max_spike_history)
        dt_all = self.current_time - connected_pre_histories

        # Apply valid spike mask to time differences
        dt_valid = self.xp.where(valid_spike_mask, dt_all, self.xp.nan)

        # STDP window masks
        # Causal window (LTP): 0 < dt < 5*tau_LTP
        ltp_mask = (dt_valid > 0) & (dt_valid < 5 * self.tau_LTP)

        # Anti-causal window (LTD): -5*tau_LTD < dt < 0
        ltd_mask = (dt_valid < 0) & (dt_valid > -5 * self.tau_LTD)

        # Calculate LTP contributions
        # Use backend.where to handle NaN values safely
        ltp_exp_terms = self.xp.where(ltp_mask,
                                self.A_LTP * self.xp.exp(-dt_valid / self.tau_LTP),
                                0.0)
        ltp_contributions = self.xp.sum(ltp_exp_terms, axis=1)

        # Calculate LTD contributions
        ltd_exp_terms = self.xp.where(ltd_mask,
                                self.A_LTD * self.xp.exp(dt_valid / self.tau_LTD),
                                0.0)
        ltd_contributions = self.xp.sum(ltd_exp_terms, axis=1)

        # Total weight changes for each connection
        total_weight_changes = ltp_contributions - ltd_contributions

        # Apply weight changes to delta_weights matrix using advanced indexing
        delta_weights[actual_post_indices, pre_indices] = total_weight_changes

        return delta_weights

    def _update_lif_layer(self, V_m, last_spike_time, refractory_until, I_ext, spike_array, layer_name):
        """
        Vectorized update of LIF neurons for one layer (backend-optimized).

        Args:
            V_m (ndarray): Membrane potentials for this layer
            last_spike_time (ndarray): Last spike times for this layer
            refractory_until (ndarray): Refractory period end times
            I_ext (ndarray): External input currents
            spike_array (ndarray): Boolean array to mark spikes
            layer_name (str): 'A' or 'B' for spike history updates

        Returns:
            ndarray: Indices of neurons that spiked
        """
        # Check which neurons are NOT in refractory period
        not_refractory = self.current_time >= refractory_until

        # For neurons in refractory period, clamp voltage to reset
        V_m[~not_refractory] = self.V_reset

        # Update membrane potential using vectorized LIF equation
        # dV/dt = (V_rest - V_m + R_m * I_ext) / tau_m
        # Only update neurons not in refractory period
        dV_dt = (self.V_rest - V_m[not_refractory] + self.R_m * I_ext[not_refractory]) / self.tau_m
        V_m[not_refractory] += dV_dt * self.dt

        # Detect spikes (threshold crossing)
        spiked = (V_m >= self.V_threshold) & not_refractory
        spiked_indices = self.xp.where(spiked)[0]

        if len(spiked_indices) > 0:
            # Reset membrane potential for spiking neurons
            V_m[spiked_indices] = self.V_reset

            # Update last spike times
            last_spike_time[spiked_indices] = self.current_time

            # Set refractory period end time
            refractory_until[spiked_indices] = self.current_time + self.refractory_period

            # Mark spikes in boolean array
            spike_array[spiked_indices] = True

            # Update spike history for STDP
            self._update_spike_history(spiked_indices, self.current_time, layer_name)

        return spiked_indices

    def update(self):
        """
        Update network for one time step using backend-accelerated operations.

        This method performs the following operations using vectorized operations:
        1. Calculate synaptic inputs for all neurons
        2. Update neuron states vectorized and detect spikes
        3. Apply STDP learning rules using vectorized calculations
        4. Clip weights to valid ranges
        5. Transfer spike indices to CPU for visualization compatibility

        Returns:
            tuple: (spiked_A_indices, spiked_B_indices) - Lists of neuron indices that spiked
        """
        self.current_time += self.dt

        # --- 1. Calculate Synaptic Inputs (backend-accelerated) ---
        I_A = self.xp.zeros(self.n_A)
        I_B = self.xp.zeros(self.n_B)

        # Add background Poisson input (increased strength for better responsiveness)
        I_A += self.poisson_input(self.background_rate_A, self.dt, self.n_A) * self.synaptic_strength * 2.0
        I_B += self.poisson_input(self.background_rate_B, self.dt, self.n_B) * self.synaptic_strength * 2.0

        # Add recurrent synaptic inputs from previous time step's spikes
        I_A += (self.W_BA @ self.spikes_B_arr) * self.synaptic_strength
        I_B += (self.W_AB @ self.spikes_A_arr) * self.synaptic_strength

        # --- 2. Backend-Accelerated Neuron Updates ---
        # Reset spike arrays
        self.spikes_A_arr.fill(False)
        self.spikes_B_arr.fill(False)

        # Layer A updates
        spiked_A_indices = self._update_lif_layer(
            self.V_m_A, self.last_spike_time_A, self.refractory_until_A,
            I_A, self.spikes_A_arr, 'A'
        )

        # Layer B updates
        spiked_B_indices = self._update_lif_layer(
            self.V_m_B, self.last_spike_time_B, self.refractory_until_B,
            I_B, self.spikes_B_arr, 'B'
        )

        # --- 3. Backend-Accelerated STDP Application ---
        # B -> A (Excitatory) connections
        if len(spiked_A_indices) > 0:
            delta_W_BA = self._vectorized_stdp_calculation(
                spiked_A_indices, 'B', 'A', self.W_BA, self.connection_mask_BA
            )
            self.W_BA += delta_W_BA

        # A -> B (Inhibitory) connections - Anti-Hebbian STDP
        if len(spiked_B_indices) > 0:
            delta_W_AB = self._vectorized_stdp_calculation(
                spiked_B_indices, 'A', 'B', self.W_AB, self.connection_mask_AB
            )
            # Anti-Hebbian STDP: Invert weight update sign for inhibitory connections
            # Pre-before-post → LTD (strengthen inhibition = more negative)
            # Post-before-pre → LTP (weaken inhibition = less negative)
            self.W_AB -= delta_W_AB

        # Clip weights to valid ranges while preserving sparse connectivity
        self.W_BA = self.xp.clip(self.W_BA, 0.0, 1.0)
        self.W_AB = self.xp.clip(self.W_AB, -1.0, 0.0)

        # Ensure non-connected positions remain at zero (preserve sparse topology)
        self.W_BA = self.xp.where(self.connection_mask_BA, self.W_BA, 0.0)
        self.W_AB = self.xp.where(self.connection_mask_AB, self.W_AB, 0.0)

        # --- 4. Critical: Transfer spike indices to CPU for visualization compatibility ---
        # Convert arrays to Python lists for API compatibility
        if self.use_gpu:
            # Transfer from GPU to CPU
            spiked_A_cpu = spiked_A_indices.get().tolist() if len(spiked_A_indices) > 0 else []
            spiked_B_cpu = spiked_B_indices.get().tolist() if len(spiked_B_indices) > 0 else []
        else:
            # Already on CPU
            spiked_A_cpu = spiked_A_indices.tolist() if len(spiked_A_indices) > 0 else []
            spiked_B_cpu = spiked_B_indices.tolist() if len(spiked_B_indices) > 0 else []

        return spiked_A_cpu, spiked_B_cpu

    # --- API Compatibility Methods ---
    def get_neuron_voltage(self, layer, neuron_idx):
        """
        Get membrane voltage for a specific neuron (for backward compatibility).

        Args:
            layer (str): 'A' or 'B'
            neuron_idx (int): Neuron index

        Returns:
            float: Membrane voltage in mV
        """
        if layer == 'A':
            voltage = self.V_m_A[neuron_idx]
        else:
            voltage = self.V_m_B[neuron_idx]

        # Transfer to CPU if needed
        if self.use_gpu:
            return float(voltage.get())
        else:
            return float(voltage)

    def get_neuron_spike_times(self, layer, neuron_idx):
        """
        Get recent spike times for a specific neuron (for backward compatibility).

        Args:
            layer (str): 'A' or 'B'
            neuron_idx (int): Neuron index

        Returns:
            np.ndarray: Array of recent spike times
        """
        if layer == 'A':
            spike_history = self.spike_history_A[neuron_idx]
        else:
            spike_history = self.spike_history_B[neuron_idx]

        # Transfer to CPU if needed and return only valid spike times (not -1000.0)
        if self.use_gpu:
            spike_history_cpu = spike_history.get()
        else:
            spike_history_cpu = spike_history

        return spike_history_cpu[spike_history_cpu > -999.0]

    def get_layer_voltages(self, layer):
        """
        Get all membrane voltages for a layer.

        Args:
            layer (str): 'A' or 'B'

        Returns:
            np.ndarray: Array of membrane voltages
        """
        if layer == 'A':
            voltages = self.V_m_A
        else:
            voltages = self.V_m_B

        # Transfer to CPU if needed
        if self.use_gpu:
            return voltages.get().copy()
        else:
            return voltages.copy()
