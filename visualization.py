"""
Visualization Module

This module contains the GUI and visualization components for the Basal Ganglia
Oscillator simulation, including real-time plotting of neural activity and
interactive controls.

Classes:
    NetworkVisualizer: Pygame-based visualization of the bipartite network
"""

import pygame
import pygame_gui
import numpy as np
from collections import deque
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from network import BipartiteNetwork


def _ensure_numpy_array(array):
    """
    Convert CuPy array to NumPy array if needed, for matplotlib compatibility.

    Args:
        array: Input array (either NumPy or CuPy)

    Returns:
        np.ndarray: NumPy array suitable for matplotlib
    """
    # Check if it's a CuPy array by looking for the .get() method
    if hasattr(array, 'get'):
        # It's a CuPy array, transfer to CPU
        return array.get()
    else:
        # It's already a NumPy array or compatible
        return array


class NetworkVisualizer:
    """
    Pygame-based visualization of the bipartite network.
    
    This class provides a comprehensive GUI for visualizing neural network activity,
    including raster plots, weight matrices, population firing rates, and interactive
    controls for simulation parameters.
    
    Attributes:
        width (int): Window width in pixels
        height (int): Window height in pixels
        screen (pygame.Surface): Main display surface
        ui_manager (pygame_gui.UIManager): GUI manager for controls
        network (BipartiteNetwork): The neural network being visualized
        spike_history_A (deque): History of spikes from Layer A
        spike_history_B (deque): History of spikes from Layer B
        rate_history_A (deque): History of firing rates for Layer A
        rate_history_B (deque): History of firing rates for Layer B
        paused (bool): Whether simulation is paused
    """
    
    def __init__(self, width=1600, height=800):
        """
        Initialize the network visualizer with specified window dimensions.
        
        Args:
            width (int, optional): Window width in pixels. Defaults to 1600.
            height (int, optional): Window height in pixels. Defaults to 800.
        """
        pygame.init()
        pygame.font.init()
        
        # --- Theme and Colors ---
        self.THEME = {
            "background": (20, 20, 30),
            "panel_bg": (40, 40, 55),
            "text": (230, 230, 230),
            "text_dark": (150, 150, 150),
            "border": (80, 80, 100),
            "grid": (60, 60, 75),
            "layer_a": (80, 120, 255),  # Layer A (inhibitory) now uses blue
            "layer_b": (255, 80, 80),   # Layer B (excitatory) now uses red
            "font_s": pygame.font.Font(None, 20),
            "font_m": pygame.font.Font(None, 24),
            "font_l": pygame.font.Font(None, 32)
        }
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bipartite Spiking Neural Network with STDP")
        
        self.ui_manager = pygame_gui.UIManager((width, height), 'theme.json')
        self.network = BipartiteNetwork(n_A=50, n_B=50)
        
        # --- Data History for Plots ---
        self.raster_window = 2000  # ms
        self.spike_history_A = deque()
        self.spike_history_B = deque()
        
        self.rate_window = 500  # ms
        self.rate_history_len = 200  # Number of points to store
        self.rate_history_A = deque(maxlen=self.rate_history_len)
        self.rate_history_B = deque(maxlen=self.rate_history_len)
        self.rate_update_interval = 50  # ms
        self.last_rate_update = 0

        self._setup_layout()
        self._create_ui_elements()
        self._create_colormaps()

        self.clock = pygame.time.Clock()
        self.paused = False

        # Track previous slider values to detect changes
        self.prev_slider_values = {}
        self._initialize_slider_tracking()

    def _setup_layout(self):
        """Define the geometry of all UI panels."""
        # Optimized layout with combined time/FPS labels to save vertical space
        self.controls_panel = pygame.Rect(10, 10, 500, 600)  # Increased width from 380 to 500
        self.rate_panel = pygame.Rect(10, 620, 380, self.height - 630)  # Expanded panel for better visualization
        self.activity_panel = pygame.Rect(400, 10, 570, self.height - 20)  # Adjusted position
        self.weights_panel = pygame.Rect(980, 10, 610, self.height - 20)

        # Sub-rects for raster plots
        self.raster_A_rect = pygame.Rect(
            self.activity_panel.x + 40,
            self.activity_panel.y + 80,
            self.activity_panel.width - 60,
            (self.activity_panel.height - 150) // 2
        )
        self.raster_B_rect = pygame.Rect(
            self.activity_panel.x + 40,
            self.raster_A_rect.bottom + 50,
            self.activity_panel.width - 60,
            (self.activity_panel.height - 150) // 2
        )

        # Sub-rects for weight matrices - ensure square proportions for 50x50 matrices
        # Reorder to match neural activity panel: inhibitory (AB) at top, excitatory (BA) at bottom
        matrix_size = 200  # Fixed size for square matrices
        self.matrix_AB_rect = pygame.Rect(
            self.weights_panel.x + 10,
            self.weights_panel.y + 60,  # Reduced from 140 to 60 to bring closer to title
            matrix_size,
            matrix_size
        )
        self.matrix_BA_rect = pygame.Rect(
            self.matrix_AB_rect.left,
            self.matrix_AB_rect.bottom + 65,  # Increased from 40 to 65 to provide space for "No Conn" legend
            matrix_size,
            matrix_size
        )

        # Bipartite graph rectangle - positioned on the right side of the weights panel
        # Adjacent to the existing heatmaps with proper spacing
        colorbar_width = 20 + 15 + 50  # colorbar + spacing + labels (from draw_weight_matrix)
        self.bipartite_graph_rect = pygame.Rect(
            self.matrix_AB_rect.right + colorbar_width + 20,  # 20px spacing from colorbar area
            self.matrix_AB_rect.y,  # Align with top matrix
            self.weights_panel.right - (self.matrix_AB_rect.right + colorbar_width + 20) - 10,  # Use remaining width
            self.matrix_BA_rect.bottom - self.matrix_AB_rect.y  # Full height of both matrices
        )

    def _create_ui_elements(self):
        """Create UI sliders, buttons, and labels."""
        # --- Labels ---
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 20, 360, 30),
            text="CONTROLS & PARAMETERS",
            manager=self.ui_manager,
            object_id='@title_label'
        )
        self.sparsity_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 70, 150, 20),
            text=f'Connect Prob: {self.network.connection_prob:.2f}',
            manager=self.ui_manager
        )
        self.rate_A_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 140, 150, 20),
            text=f'Spont Rate A: {self.network.background_rate_A:.1f} Hz',
            manager=self.ui_manager
        )
        self.rate_B_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 210, 150, 20),
            text=f'Spont Rate B: {self.network.background_rate_B:.1f} Hz',
            manager=self.ui_manager
        )

        # --- Lognormal Distribution Labels ---
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 280, 360, 25),
            text="LOGNORMAL WEIGHT DISTRIBUTION",
            manager=self.ui_manager,
            object_id='@section_label'
        )

        # W_BA (Excitatory) parameters
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 315, 170, 20),
            text="W_BA (Excitatory):",
            manager=self.ui_manager,
            object_id='@subsection_label'
        )
        self.mu_BA_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 345, 170, 20),
            text=f'μ (Mean): {self.network.lognorm_mu_BA:.2f}',
            manager=self.ui_manager
        )
        self.sigma_BA_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 395, 170, 20),
            text=f'σ (Std Dev): {self.network.lognorm_sigma_BA:.2f}',
            manager=self.ui_manager
        )
        self.scale_BA_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 445, 170, 20),
            text=f'Scale: {self.network.lognorm_scale_BA:.3f}',
            manager=self.ui_manager
        )

        # W_AB (Inhibitory) parameters
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(200, 315, 170, 20),
            text="W_AB (Inhibitory):",
            manager=self.ui_manager,
            object_id='@subsection_label'
        )
        self.mu_AB_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(200, 345, 170, 20),
            text=f'μ (Mean): {self.network.lognorm_mu_AB:.2f}',
            manager=self.ui_manager
        )
        self.sigma_AB_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(200, 395, 170, 20),
            text=f'σ (Std Dev): {self.network.lognorm_sigma_AB:.2f}',
            manager=self.ui_manager
        )
        self.scale_AB_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(200, 445, 170, 20),
            text=f'Scale: {self.network.lognorm_scale_AB:.3f}',
            manager=self.ui_manager
        )

        self.time_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 580, 180, 20),
            text="Time: 0.00 s",
            manager=self.ui_manager
        )
        self.fps_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(210, 580, 90, 20),
            text="FPS: 0",
            manager=self.ui_manager
        )

        # --- Sliders ---
        self.sparsity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 100, 180, 25),
            start_value=self.network.connection_prob,
            value_range=(0.0, 1.0),
            click_increment=0.01,  # 1% increments for connection probability
            manager=self.ui_manager
        )
        self.rate_A_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 170, 180, 25),
            start_value=self.network.background_rate_A,
            value_range=(0.0, 50.0),
            click_increment=0.5,  # 0.5 Hz increments for background rates
            manager=self.ui_manager
        )
        self.rate_B_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 240, 180, 25),
            start_value=self.network.background_rate_B,
            value_range=(0.0, 50.0),
            click_increment=0.5,  # 0.5 Hz increments for background rates
            manager=self.ui_manager
        )

        # --- STDP Rules Section ---
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(220, 70, 180, 25),
            text="STDP RULES",
            manager=self.ui_manager,
            object_id='@section_label'
        )

        self.estdp_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(220, 100, 180, 30),
            text='eSTDP: OFF',
            manager=self.ui_manager,
            object_id='#estdp_button'
        )

        self.hebbian_istdp_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(220, 140, 180, 30),
            text='iSTDP Hebbian: OFF',
            manager=self.ui_manager,
            object_id='#hebbian_istdp_button'
        )

        self.anti_hebbian_istdp_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(220, 180, 180, 30),
            text='iSTDP Anti-Hebb: OFF',
            manager=self.ui_manager,
            object_id='#anti_hebbian_istdp_button'
        )

        # --- Lognormal Parameter Sliders ---
        # W_BA (Excitatory) sliders
        self.mu_BA_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 370, 170, 20),
            start_value=self.network.lognorm_mu_BA,
            value_range=(-3.0, 1.0),
            click_increment=0.05,  # 0.05 increments for mu (range: 4.0, so ~80 steps)
            manager=self.ui_manager
        )
        self.sigma_BA_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 420, 170, 20),
            start_value=self.network.lognorm_sigma_BA,
            value_range=(0.1, 2.0),
            click_increment=0.02,  # 0.02 increments for sigma (range: 1.9, so ~95 steps)
            manager=self.ui_manager
        )
        self.scale_BA_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(20, 470, 170, 20),
            start_value=self.network.lognorm_scale_BA,
            value_range=(0.01, 1.0),
            click_increment=0.01,  # 0.01 increments for scale (range: 0.99, so ~99 steps)
            manager=self.ui_manager
        )

        # W_AB (Inhibitory) sliders
        self.mu_AB_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(200, 370, 170, 20),
            start_value=self.network.lognorm_mu_AB,
            value_range=(-3.0, 1.0),
            click_increment=0.05,  # 0.05 increments for mu (range: 4.0, so ~80 steps)
            manager=self.ui_manager
        )
        self.sigma_AB_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(200, 420, 170, 20),
            start_value=self.network.lognorm_sigma_AB,
            value_range=(0.1, 2.0),
            click_increment=0.02,  # 0.02 increments for sigma (range: 1.9, so ~95 steps)
            manager=self.ui_manager
        )
        self.scale_AB_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(200, 470, 170, 20),
            start_value=self.network.lognorm_scale_AB,
            value_range=(0.01, 1.0),
            click_increment=0.01,  # 0.01 increments for scale (range: 0.99, so ~99 steps)
            manager=self.ui_manager
        )

        # --- Buttons ---
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(20, 500, 110, 35),
            text='Reset',
            manager=self.ui_manager
        )
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(140, 500, 110, 35),
            text='Pause',
            manager=self.ui_manager
        )
        self.regenerate_weights_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(260, 500, 110, 35),
            text='Regen. W',
            manager=self.ui_manager
        )

    def _create_colormaps(self):
        """Generate RGB arrays for colormaps to be used in heatmaps."""
        self.cmap_hot = cm.get_cmap('hot')
        self.cmap_cool_r = cm.get_cmap('cool_r')  # Reversed cool colormap for inhibitory weights

    def _initialize_slider_tracking(self):
        """Initialize tracking of slider values to detect changes."""
        # This will be called after sliders are created in _create_ui_elements
        # Initialize with current values to avoid false positives on first check
        if hasattr(self, 'sparsity_slider'):
            self.prev_slider_values = {
                'sparsity': self.sparsity_slider.get_current_value(),
                'rate_A': self.rate_A_slider.get_current_value(),
                'rate_B': self.rate_B_slider.get_current_value(),
                'mu_BA': self.mu_BA_slider.get_current_value(),
                'sigma_BA': self.sigma_BA_slider.get_current_value(),
                'scale_BA': self.scale_BA_slider.get_current_value(),
                'mu_AB': self.mu_AB_slider.get_current_value(),
                'sigma_AB': self.sigma_AB_slider.get_current_value(),
                'scale_AB': self.scale_AB_slider.get_current_value(),
            }



    def _draw_panel(self, rect, title):
        """Helper to draw a styled panel background and title."""
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], rect, border_radius=5)
        pygame.draw.rect(self.screen, self.THEME["border"], rect, 1, border_radius=5)
        title_surf = self.THEME["font_m"].render(title, True, self.THEME["text"])
        self.screen.blit(title_surf, (rect.x + 10, rect.y + 10))

    def draw_raster_plot(self, rect, spike_history, n_neurons, color, title):
        """Draw a rich raster plot for a neural population."""
        # --- Axis labels ---
        y_axis_label = self.THEME["font_s"].render("Neuron ID", True, self.THEME["text_dark"])
        self.screen.blit(
            pygame.transform.rotate(y_axis_label, 90),
            (rect.x - 30, rect.centery - y_axis_label.get_width()//2)
        )

        # --- Plot Title ---
        title_surf = self.THEME["font_m"].render(title, True, color)
        self.screen.blit(title_surf, (rect.x, rect.y - 30))

        # --- Draw complete grid with border ---
        # First draw the rectangular border around the entire plot area
        pygame.draw.rect(self.screen, self.THEME["grid"], rect, 1)

        # Draw horizontal grid lines (neuron divisions)
        for i in range(0, n_neurons, 20):
            if i > 0:  # Skip the top border (already drawn)
                y = rect.y + (i / n_neurons) * rect.height
                pygame.draw.line(self.screen, self.THEME["grid"], (rect.x, y), (rect.right, y), 1)

        # Draw vertical grid lines (time divisions)
        current_time = self.network.current_time
        start_time = max(0, current_time - self.raster_window)
        for i in range(1, 5):  # Skip first and last (borders already drawn)
            t = start_time + i * self.raster_window / 4
            x = rect.x + (t - start_time) / self.raster_window * rect.width
            pygame.draw.line(self.screen, self.THEME["grid"], (x, rect.top), (x, rect.bottom), 1)

        # Draw time axis labels
        for i in range(5):
            t = start_time + i * self.raster_window / 4
            x = rect.x + (t - start_time) / self.raster_window * rect.width
            time_text = self.THEME["font_s"].render(f"{t/1000:.1f}s", True, self.THEME["text_dark"])
            self.screen.blit(time_text, (x - time_text.get_width()//2, rect.bottom + 5))

        # --- Draw Spikes ---
        for spike_time, neuron_id in spike_history:
            if start_time <= spike_time <= current_time:
                x = rect.x + ((spike_time - start_time) / self.raster_window) * rect.width
                y = rect.y + (neuron_id / n_neurons) * rect.height
                if rect.collidepoint(x, y):
                     pygame.draw.line(self.screen, color, (x, y), (x, y + 2), 2)

    def draw_weight_matrix(self, rect, weights, cmap, title, clim):
        """Draw a heatmap of a weight matrix with distinct colors for non-existent connections."""
        # Convert CuPy array to NumPy array for matplotlib compatibility
        weights_cpu = _ensure_numpy_array(weights)

        if weights_cpu.size == 0:
            return

        title_surf = self.THEME["font_m"].render(title, True, self.THEME["text"])
        self.screen.blit(title_surf, (rect.x, rect.y - 30))

        # Create mask for non-existent connections (weight = 0)
        connection_mask = weights_cpu != 0

        # Define color for non-existent connections (neutral gray)
        no_connection_color = np.array([0.3, 0.3, 0.3])  # Dark gray RGB

        # Normalize weights for colormap (only for existing connections)
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

        # Initialize RGB array with no-connection color
        rgb_array = np.full((*weights_cpu.T.shape, 3), no_connection_color)

        # Apply colormap only to positions with actual connections
        if np.any(connection_mask):
            # Get RGBA array from colormap for existing connections
            rgba_array = cmap(norm(weights_cpu.T))
            # Update RGB array only where connections exist
            rgb_array[connection_mask.T] = rgba_array[connection_mask.T, :3]

        # Convert to 8-bit RGB
        rgb_array = (rgb_array * 255).astype(np.uint8)

        # Create surface, blit the array, and scale to fit the rect
        matrix_surf = pygame.Surface(rgb_array.shape[:2])
        pygame.surfarray.blit_array(matrix_surf, rgb_array)
        scaled_surf = pygame.transform.scale(matrix_surf, (rect.width, rect.height))
        self.screen.blit(scaled_surf, rect)
        pygame.draw.rect(self.screen, self.THEME["border"], rect, 1)

        # --- Draw Colorbar ---
        cbar_rect = pygame.Rect(rect.right + 15, rect.y, 20, rect.height)
        for i in range(cbar_rect.height):
            # Interpolate color from bottom to top
            val = clim[0] + (clim[1] - clim[0]) * (1 - i / cbar_rect.height)
            color = cmap(norm(val))
            pygame.draw.line(
                self.screen,
                [c*255 for c in color[:3]],
                (cbar_rect.left, cbar_rect.y + i),
                (cbar_rect.right, cbar_rect.y+i),
                1
            )

        max_label = self.THEME["font_s"].render(f"{clim[1]:.1f}", True, self.THEME["text"])
        min_label = self.THEME["font_s"].render(f"{clim[0]:.1f}", True, self.THEME["text"])
        self.screen.blit(max_label, (cbar_rect.right + 5, cbar_rect.top - max_label.get_height()//2))
        self.screen.blit(min_label, (cbar_rect.right + 5, cbar_rect.bottom - min_label.get_height()//2))

        # Add legend for no-connection color
        no_conn_label = self.THEME["font_s"].render("No Conn", True, self.THEME["text"])
        no_conn_rect = pygame.Rect(cbar_rect.left - 10, cbar_rect.bottom + 10, 15, 15)
        pygame.draw.rect(self.screen, [c*255 for c in no_connection_color], no_conn_rect)
        pygame.draw.rect(self.screen, self.THEME["border"], no_conn_rect, 1)
        self.screen.blit(no_conn_label, (no_conn_rect.right + 5, no_conn_rect.y))

    def draw_bipartite_graph(self, rect, W_AB, W_BA, title):
        """
        Draw a split bipartite graph visualization of synaptic weights.

        Displays inhibitory and excitatory connections in separate subpanels
        for improved readability and analysis.

        Args:
            rect (pygame.Rect): Rectangle to draw the graph within
            W_AB (array): Inhibitory weight matrix (A -> B)
            W_BA (array): Excitatory weight matrix (B -> A)
            title (str): Title to display at the top of the graph
        """
        # Convert CuPy arrays to NumPy arrays for processing
        W_AB_cpu = _ensure_numpy_array(W_AB)
        W_BA_cpu = _ensure_numpy_array(W_BA)

        if W_AB_cpu.size == 0 or W_BA_cpu.size == 0:
            return

        # Draw main title
        title_surf = self.THEME["font_m"].render(title, True, self.THEME["text"])
        self.screen.blit(title_surf, (rect.x, rect.y - 30))

        # Define common parameters
        n_A = self.network.n_A  # A neurons (inhibitory)
        n_B = self.network.n_B  # B neurons (excitatory)
        node_radius = 3
        scale_factor = 2.5
        max_line_width = 2

        # Calculate dynamic thresholds for each connection type using count-based approach
        # Inhibitory threshold from W_AB_cpu matrix
        inhibitory_weights = W_AB_cpu[W_AB_cpu != 0]  # Extract non-zero values
        if len(inhibitory_weights) > 0:
            abs_weights = np.abs(inhibitory_weights)  # Take absolute values (inhibitory weights may be negative)
            sorted_weights = np.sort(abs_weights)[::-1]  # Sort in descending order
            num_connections = len(sorted_weights)
            top_10_percent_count = int(np.ceil(0.1 * num_connections))  # Exactly top 10% count
            if top_10_percent_count > 0:
                cutoff_weight = sorted_weights[top_10_percent_count - 1]  # Weight of weakest in top 10%
                inhibitory_threshold = cutoff_weight - 1e-9  # Slightly below to ensure strict > works
            else:
                inhibitory_threshold = float('inf')  # Edge case: no connections to show
        else:
            inhibitory_threshold = float('inf')  # No connections exist, prevent any drawing

        # Excitatory threshold from W_BA_cpu matrix
        excitatory_weights = W_BA_cpu[W_BA_cpu != 0]  # Extract non-zero values
        if len(excitatory_weights) > 0:
            sorted_weights = np.sort(excitatory_weights)[::-1]  # Sort in descending order
            num_connections = len(sorted_weights)
            top_10_percent_count = int(np.ceil(0.1 * num_connections))  # Exactly top 10% count
            if top_10_percent_count > 0:
                cutoff_weight = sorted_weights[top_10_percent_count - 1]  # Weight of weakest in top 10%
                excitatory_threshold = cutoff_weight - 1e-9  # Slightly below to ensure strict > works
            else:
                excitatory_threshold = float('inf')  # Edge case: no connections to show
        else:
            excitatory_threshold = float('inf')  # No connections exist, prevent any drawing

        # Split the main rectangle into two subpanels
        subpanel_spacing = 10
        subpanel_height = (rect.height - subpanel_spacing) // 2

        # Top subpanel: Inhibitory connections (A -> B)
        inhibitory_rect = pygame.Rect(rect.x, rect.y, rect.width, subpanel_height)

        # Bottom subpanel: Excitatory connections (B -> A)
        excitatory_rect = pygame.Rect(rect.x, rect.y + subpanel_height + subpanel_spacing,
                                     rect.width, subpanel_height)

        # Draw inhibitory connections subpanel
        self._draw_inhibitory_subpanel(inhibitory_rect, W_AB_cpu, n_A, n_B,
                                     node_radius, scale_factor, max_line_width, inhibitory_threshold)

        # Draw excitatory connections subpanel
        self._draw_excitatory_subpanel(excitatory_rect, W_BA_cpu, n_A, n_B,
                                     node_radius, scale_factor, max_line_width, excitatory_threshold)

    def _draw_inhibitory_subpanel(self, rect, W_AB_cpu, n_A, n_B, node_radius,
                                 scale_factor, max_line_width, threshold):
        """
        Draw the inhibitory connections subpanel (A -> B).

        Shows Layer A neurons on the left, Layer B neurons on the right,
        with blue inhibitory connection lines.
        """
        # Draw subpanel title
        title_surf = self.THEME["font_s"].render("Inhibitory Connections (A -> B)",
                                                True, self.THEME["layer_a"])
        title_pos = (rect.x + 5, rect.y + 2)
        self.screen.blit(title_surf, title_pos)

        # Calculate node positions
        left_margin = 20
        right_margin = 20
        layer_A_x = rect.x + left_margin
        layer_B_x = rect.right - right_margin

        # Distribute nodes evenly across height with padding for title
        top_padding = 25  # Extra space for title
        bottom_padding = 10
        available_height = rect.height - top_padding - bottom_padding

        # Calculate Layer A positions (left side)
        layer_A_positions = []
        for i in range(n_A):
            if n_A > 1:
                y = rect.y + top_padding + (i / (n_A - 1)) * available_height
            else:
                y = rect.y + rect.height // 2
            layer_A_positions.append((layer_A_x, int(y)))

        # Calculate Layer B positions (right side)
        layer_B_positions = []
        for i in range(n_B):
            if n_B > 1:
                y = rect.y + top_padding + (i / (n_B - 1)) * available_height
            else:
                y = rect.y + rect.height // 2
            layer_B_positions.append((layer_B_x, int(y)))

        # Create transparent surface for connections
        connection_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        connection_surface.fill((0, 0, 0, 0))

        # Draw inhibitory connections (A -> B)
        for i in range(n_A):  # Source: A neurons
            for j in range(n_B):  # Target: B neurons
                weight = W_AB_cpu[j, i]
                if abs(weight) > threshold:
                    line_width = min(max_line_width, max(1, int(1 + abs(weight) * scale_factor)))
                    start_pos = (layer_A_positions[i][0] - rect.x,
                                layer_A_positions[i][1] - rect.y)
                    end_pos = (layer_B_positions[j][0] - rect.x,
                              layer_B_positions[j][1] - rect.y)
                    # Blue color for inhibitory connections
                    color_with_alpha = (*self.THEME["layer_a"], 180)
                    pygame.draw.line(connection_surface, color_with_alpha, start_pos, end_pos, line_width)

        # Blit connections to screen
        self.screen.blit(connection_surface, (rect.x, rect.y))

        # Draw nodes on top of connections
        # Layer A nodes (inhibitory) - left side
        for pos in layer_A_positions:
            pygame.draw.circle(self.screen, self.THEME["layer_a"], pos, node_radius)
            pygame.draw.circle(self.screen, self.THEME["border"], pos, node_radius, 1)

        # Layer B nodes (excitatory) - right side
        for pos in layer_B_positions:
            pygame.draw.circle(self.screen, self.THEME["layer_b"], pos, node_radius)
            pygame.draw.circle(self.screen, self.THEME["border"], pos, node_radius, 1)

        # Add layer labels
        layer_a_label = self.THEME["font_s"].render("Layer A", True, self.THEME["layer_a"])
        layer_b_label = self.THEME["font_s"].render("Layer B", True, self.THEME["layer_b"])

        # Layer A label (bottom-left)
        layer_a_label_pos = (layer_A_x - layer_a_label.get_width() // 2,
                            rect.bottom - 20)
        layer_a_bg = pygame.Rect(layer_a_label_pos[0] - 2, layer_a_label_pos[1] - 2,
                                layer_a_label.get_width() + 4, layer_a_label.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], layer_a_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], layer_a_bg, 1)
        self.screen.blit(layer_a_label, layer_a_label_pos)

        # Layer B label (bottom-right)
        layer_b_label_pos = (layer_B_x - layer_b_label.get_width() // 2,
                            rect.bottom - 20)
        layer_b_bg = pygame.Rect(layer_b_label_pos[0] - 2, layer_b_label_pos[1] - 2,
                                layer_b_label.get_width() + 4, layer_b_label.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], layer_b_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], layer_b_bg, 1)
        self.screen.blit(layer_b_label, layer_b_label_pos)

        # Draw border around subpanel
        pygame.draw.rect(self.screen, self.THEME["border"], rect, 1)

    def _draw_excitatory_subpanel(self, rect, W_BA_cpu, n_A, n_B, node_radius,
                                 scale_factor, max_line_width, threshold):
        """
        Draw the excitatory connections subpanel (B -> A).

        Shows Layer B neurons on the left, Layer A neurons on the right,
        with red excitatory connection lines.
        """
        # Draw subpanel title
        title_surf = self.THEME["font_s"].render("Excitatory Connections (B -> A)",
                                                True, self.THEME["layer_b"])
        title_pos = (rect.x + 5, rect.y + 2)
        self.screen.blit(title_surf, title_pos)

        # Calculate node positions
        left_margin = 20
        right_margin = 20
        layer_B_x = rect.x + left_margin  # B neurons on left for B->A connections
        layer_A_x = rect.right - right_margin  # A neurons on right

        # Distribute nodes evenly across height with padding for title
        top_padding = 25  # Extra space for title
        bottom_padding = 10
        available_height = rect.height - top_padding - bottom_padding

        # Calculate Layer B positions (left side)
        layer_B_positions = []
        for i in range(n_B):
            if n_B > 1:
                y = rect.y + top_padding + (i / (n_B - 1)) * available_height
            else:
                y = rect.y + rect.height // 2
            layer_B_positions.append((layer_B_x, int(y)))

        # Calculate Layer A positions (right side)
        layer_A_positions = []
        for i in range(n_A):
            if n_A > 1:
                y = rect.y + top_padding + (i / (n_A - 1)) * available_height
            else:
                y = rect.y + rect.height // 2
            layer_A_positions.append((layer_A_x, int(y)))

        # Create transparent surface for connections
        connection_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        connection_surface.fill((0, 0, 0, 0))

        # Draw excitatory connections (B -> A)
        for i in range(n_B):  # Source: B neurons
            for j in range(n_A):  # Target: A neurons
                weight = W_BA_cpu[j, i]
                if abs(weight) > threshold:
                    line_width = min(max_line_width, max(1, int(1 + abs(weight) * scale_factor)))
                    start_pos = (layer_B_positions[i][0] - rect.x,
                                layer_B_positions[i][1] - rect.y)
                    end_pos = (layer_A_positions[j][0] - rect.x,
                              layer_A_positions[j][1] - rect.y)
                    # Red color for excitatory connections
                    color_with_alpha = (*self.THEME["layer_b"], 180)
                    pygame.draw.line(connection_surface, color_with_alpha, start_pos, end_pos, line_width)

        # Blit connections to screen
        self.screen.blit(connection_surface, (rect.x, rect.y))

        # Draw nodes on top of connections
        # Layer B nodes (excitatory) - left side
        for pos in layer_B_positions:
            pygame.draw.circle(self.screen, self.THEME["layer_b"], pos, node_radius)
            pygame.draw.circle(self.screen, self.THEME["border"], pos, node_radius, 1)

        # Layer A nodes (inhibitory) - right side
        for pos in layer_A_positions:
            pygame.draw.circle(self.screen, self.THEME["layer_a"], pos, node_radius)
            pygame.draw.circle(self.screen, self.THEME["border"], pos, node_radius, 1)

        # Add layer labels
        layer_b_label = self.THEME["font_s"].render("Layer B", True, self.THEME["layer_b"])
        layer_a_label = self.THEME["font_s"].render("Layer A", True, self.THEME["layer_a"])

        # Layer B label (bottom-left)
        layer_b_label_pos = (layer_B_x - layer_b_label.get_width() // 2,
                            rect.bottom - 20)
        layer_b_bg = pygame.Rect(layer_b_label_pos[0] - 2, layer_b_label_pos[1] - 2,
                                layer_b_label.get_width() + 4, layer_b_label.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], layer_b_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], layer_b_bg, 1)
        self.screen.blit(layer_b_label, layer_b_label_pos)

        # Layer A label (bottom-right)
        layer_a_label_pos = (layer_A_x - layer_a_label.get_width() // 2,
                            rect.bottom - 20)
        layer_a_bg = pygame.Rect(layer_a_label_pos[0] - 2, layer_a_label_pos[1] - 2,
                                layer_a_label.get_width() + 4, layer_a_label.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], layer_a_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], layer_a_bg, 1)
        self.screen.blit(layer_a_label, layer_a_label_pos)

        # Draw border around subpanel
        pygame.draw.rect(self.screen, self.THEME["border"], rect, 1)

    def calculate_and_update_rates(self):
        """Calculate population firing rates and store in history."""
        if self.network.current_time - self.last_rate_update < self.rate_update_interval:
            return
        self.last_rate_update = self.network.current_time

        rate_A = sum(1 for t, _ in self.spike_history_A if self.network.current_time - t < self.rate_window)
        rate_B = sum(1 for t, _ in self.spike_history_B if self.network.current_time - t < self.rate_window)

        # Normalize by window and number of neurons
        norm_factor = (self.rate_window / 1000.0)
        avg_rate_A = (rate_A / self.network.n_A) / norm_factor if self.network.n_A > 0 else 0
        avg_rate_B = (rate_B / self.network.n_B) / norm_factor if self.network.n_B > 0 else 0

        self.rate_history_A.append(avg_rate_A)
        self.rate_history_B.append(avg_rate_B)

    def draw_rate_plots(self, base_rect):
        """Draw time-series plots of population firing rates."""
        # Use the full rate panel area with some padding
        plot_rect = pygame.Rect(base_rect.x + 10, base_rect.y + 10, base_rect.width - 20, base_rect.height - 20)
        self._draw_panel(plot_rect, "Population Firing Rates (Hz)")

        # Draw plot area with more space for time axis labels
        graph_rect = pygame.Rect(plot_rect.x + 40, plot_rect.y + 30, plot_rect.width - 50, plot_rect.height - 60)

        # Determine max rate for y-axis scaling, with dynamic minimum
        all_rates = []
        if self.rate_history_A:
            all_rates.extend(self.rate_history_A)
        if self.rate_history_B:
            all_rates.extend(self.rate_history_B)

        if all_rates:
            min_rate = min(all_rates)
            max_rate = max(all_rates)
            # Add some padding to the range
            rate_range = max_rate - min_rate
            if rate_range < 1.0:  # Minimum range for visibility
                rate_range = 1.0
                max_rate = min_rate + rate_range
            padding = rate_range * 0.1
            min_rate = max(0, min_rate - padding)  # Don't go below 0
            max_rate = max_rate + padding
        else:
            min_rate = 0.0
            max_rate = 10.0

        # --- Draw axes and grid ---
        # Y-axis
        pygame.draw.line(
            self.screen,
            self.THEME["border"],
            (graph_rect.left, graph_rect.top),
            (graph_rect.left, graph_rect.bottom),
            2
        )
        # X-axis
        pygame.draw.line(
            self.screen,
            self.THEME["border"],
            (graph_rect.left, graph_rect.bottom),
            (graph_rect.right, graph_rect.bottom),
            2
        )

        # Y-axis labels and horizontal grid lines
        for i in range(5):
            y_val = min_rate + (max_rate - min_rate) * i / 4
            y_pos = graph_rect.bottom - (i / 4) * graph_rect.height

            # Grid line
            pygame.draw.line(
                self.screen,
                self.THEME["grid"],
                (graph_rect.left, y_pos),
                (graph_rect.right, y_pos),
                1
            )

            # Y-axis label
            y_label = self.THEME["font_s"].render(f"{y_val:.1f}", True, self.THEME["text_dark"])
            self.screen.blit(y_label, (graph_rect.left - 35, y_pos - y_label.get_height()//2))

        # Time axis labels and vertical grid lines
        current_time = self.network.current_time
        time_window = self.rate_history_len * self.rate_update_interval  # Total time window in ms
        start_time = max(0, current_time - time_window)

        for i in range(5):
            time_val = start_time + (time_window * i / 4)
            x_pos = graph_rect.left + (i / 4) * graph_rect.width

            # Vertical grid line
            pygame.draw.line(
                self.screen,
                self.THEME["grid"],
                (x_pos, graph_rect.top),
                (x_pos, graph_rect.bottom),
                1
            )

            # Time label
            time_label = self.THEME["font_s"].render(f"{time_val/1000:.1f}s", True, self.THEME["text_dark"])
            self.screen.blit(time_label, (x_pos - time_label.get_width()//2, graph_rect.bottom + 5))

        # --- Plot data ---
        def plot_line(history, color):
            if len(history) > 1:
                points = []
                for i, rate in enumerate(history):
                    x = graph_rect.x + (i / (self.rate_history_len-1)) * graph_rect.width
                    y = graph_rect.bottom - ((rate - min_rate) / (max_rate - min_rate)) * graph_rect.height
                    points.append((x, y))
                pygame.draw.lines(self.screen, color, False, points, 2)

        plot_line(self.rate_history_A, self.THEME["layer_a"])
        plot_line(self.rate_history_B, self.THEME["layer_b"])

        # Add legend outside the plot area with background for better visibility
        legend_x = plot_rect.right + 5  # Position outside the plot area
        legend_y = plot_rect.y + 35

        # Create background rectangles for better contrast
        legend_a = self.THEME["font_s"].render("Layer A", True, self.THEME["layer_a"])
        legend_b = self.THEME["font_s"].render("Layer B", True, self.THEME["layer_b"])

        # Background rectangle for Layer A label
        legend_a_bg = pygame.Rect(legend_x - 2, legend_y - 2,
                                 legend_a.get_width() + 4, legend_a.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], legend_a_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], legend_a_bg, 1)

        # Background rectangle for Layer B label
        legend_b_bg = pygame.Rect(legend_x - 2, legend_y + 15 - 2,
                                 legend_b.get_width() + 4, legend_b.get_height() + 4)
        pygame.draw.rect(self.screen, self.THEME["panel_bg"], legend_b_bg)
        pygame.draw.rect(self.screen, self.THEME["border"], legend_b_bg, 1)

        # Draw the labels on top of backgrounds
        self.screen.blit(legend_a, (legend_x, legend_y))
        self.screen.blit(legend_b, (legend_x, legend_y + 15))

    def handle_events(self, event):
        """Handle pygame and pygame_gui events."""
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.sparsity_slider:
                    # Update connection probability and regenerate connectivity in real-time
                    self.network.connection_prob = event.value
                    self.network.initialize_connections()
                    self.sparsity_label.set_text(f'Connect Prob: {event.value:.2f}')
                elif event.ui_element == self.rate_A_slider:
                    self.network.background_rate_A = event.value
                    self.rate_A_label.set_text(f'Spont Rate A: {event.value:.1f} Hz')
                elif event.ui_element == self.rate_B_slider:
                    self.network.background_rate_B = event.value
                    self.rate_B_label.set_text(f'Spont Rate B: {event.value:.1f} Hz')

                # Handle lognormal parameter sliders for W_BA (excitatory)
                elif event.ui_element == self.mu_BA_slider:
                    self.network.update_lognormal_parameters(mu_BA=event.value)
                    self.mu_BA_label.set_text(f'μ (Mean): {event.value:.2f}')
                elif event.ui_element == self.sigma_BA_slider:
                    self.network.update_lognormal_parameters(sigma_BA=event.value)
                    self.sigma_BA_label.set_text(f'σ (Std Dev): {event.value:.2f}')
                elif event.ui_element == self.scale_BA_slider:
                    self.network.update_lognormal_parameters(scale_BA=event.value)
                    self.scale_BA_label.set_text(f'Scale: {event.value:.3f}')

                # Handle lognormal parameter sliders for W_AB (inhibitory)
                elif event.ui_element == self.mu_AB_slider:
                    self.network.update_lognormal_parameters(mu_AB=event.value)
                    self.mu_AB_label.set_text(f'μ (Mean): {event.value:.2f}')
                elif event.ui_element == self.sigma_AB_slider:
                    self.network.update_lognormal_parameters(sigma_AB=event.value)
                    self.sigma_AB_label.set_text(f'σ (Std Dev): {event.value:.2f}')
                elif event.ui_element == self.scale_AB_slider:
                    self.network.update_lognormal_parameters(scale_AB=event.value)
                    self.scale_AB_label.set_text(f'Scale: {event.value:.3f}')

            elif event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.reset_button:
                    # Update connection probability from slider
                    self.network.connection_prob = self.sparsity_slider.get_current_value()
                    # Reinitialize connectivity with new probability
                    self.network.initialize_connections()
                    # Reset all neuron states and simulation time
                    self.network.reset_network_state()
                    # Clear visualization history
                    self.spike_history_A.clear()
                    self.spike_history_B.clear()
                    self.rate_history_A.clear()
                    self.rate_history_B.clear()
                    # Reset rate calculation timing
                    self.last_rate_update = 0
                elif event.ui_element == self.pause_button:
                    self.paused = not self.paused
                    self.pause_button.set_text('Resume' if self.paused else 'Pause')
                elif event.ui_element == self.regenerate_weights_button:
                    # Regenerate weights with current lognormal parameters
                    self.network.initialize_connections()

                elif event.ui_element == self.estdp_button:
                    # Toggle excitatory STDP on/off
                    self.network.eSTDP_enabled = not self.network.eSTDP_enabled
                    self.estdp_button.set_text(f"eSTDP: {'ON' if self.network.eSTDP_enabled else 'OFF'}")

                elif event.ui_element == self.hebbian_istdp_button:
                    # Toggle Hebbian iSTDP rule
                    if self.network.iSTDP_enabled and not self.network.is_iSTDP_anti_hebbian:
                        # Currently Hebbian ON -> Turn OFF (disable all iSTDP)
                        self.network.iSTDP_enabled = False
                        self.hebbian_istdp_button.set_text('iSTDP Hebbian: OFF')
                        self.anti_hebbian_istdp_button.set_text('iSTDP Anti-Hebb: OFF')
                    else:
                        # Currently OFF or Anti-Hebbian ON -> Turn Hebbian ON
                        self.network.iSTDP_enabled = True
                        self.network.is_iSTDP_anti_hebbian = False
                        self.hebbian_istdp_button.set_text('iSTDP Hebbian: ON')
                        self.anti_hebbian_istdp_button.set_text('iSTDP Anti-Hebb: OFF')

                elif event.ui_element == self.anti_hebbian_istdp_button:
                    # Toggle Anti-Hebbian iSTDP rule
                    if self.network.iSTDP_enabled and self.network.is_iSTDP_anti_hebbian:
                        # Currently Anti-Hebbian ON -> Turn OFF (disable all iSTDP)
                        self.network.iSTDP_enabled = False
                        self.hebbian_istdp_button.set_text('iSTDP Hebbian: OFF')
                        self.anti_hebbian_istdp_button.set_text('iSTDP Anti-Hebb: OFF')
                    else:
                        # Currently OFF or Hebbian ON -> Turn Anti-Hebbian ON
                        self.network.iSTDP_enabled = True
                        self.network.is_iSTDP_anti_hebbian = True
                        self.anti_hebbian_istdp_button.set_text('iSTDP Anti-Hebb: ON')
                        self.hebbian_istdp_button.set_text('iSTDP Hebbian: OFF')

    def run(self):
        """Main simulation loop."""
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_events(event)
                self.ui_manager.process_events(event)

            # --- Update Simulation ---
            if not self.paused:
                # Run multiple simulation steps per frame for speed
                for _ in range(5):
                    spikes_A, spikes_B = self.network.update()
                    current_time = self.network.current_time
                    # Record spikes with a cap to prevent performance degradation
                    if len(self.spike_history_A) > 5000:
                        self.spike_history_A.popleft()
                    if len(self.spike_history_B) > 5000:
                        self.spike_history_B.popleft()
                    for neuron_id in spikes_A:
                        self.spike_history_A.append((current_time, neuron_id))
                    for neuron_id in spikes_B:
                        self.spike_history_B.append((current_time, neuron_id))

                self.calculate_and_update_rates()

            # --- Update UI and Labels ---
            self.ui_manager.update(time_delta)
            self.time_label.set_text(f"Time: {self.network.current_time/1000:.2f} s")
            self.fps_label.set_text(f"FPS: {self.clock.get_fps():.0f}")



            # --- Draw Everything ---
            self.screen.fill(self.THEME["background"])

            # Draw Panels
            self._draw_panel(self.controls_panel, "")
            self._draw_panel(self.rate_panel, "")
            self._draw_panel(self.activity_panel, "NEURAL ACTIVITY")
            self._draw_panel(self.weights_panel, "SYNAPTIC WEIGHTS")

            # Draw Plots
            self.draw_raster_plot(
                self.raster_A_rect,
                self.spike_history_A,
                self.network.n_A,
                self.THEME["layer_a"],
                "Layer A (Inhibitory)"
            )
            self.draw_raster_plot(
                self.raster_B_rect,
                self.spike_history_B,
                self.network.n_B,
                self.THEME["layer_b"],
                "Layer B (Excitatory)"
            )

            self.draw_weight_matrix(
                self.matrix_AB_rect,
                self.network.W_AB,
                self.cmap_cool_r,
                "Inhibitory Weights (A -> B)",
                (-1.0, 0.0)
            )
            self.draw_weight_matrix(
                self.matrix_BA_rect,
                self.network.W_BA,
                self.cmap_hot,
                "Excitatory Weights (B -> A)",
                (0.0, 1.0)
            )

            # Draw bipartite graph visualization
            self.draw_bipartite_graph(
                self.bipartite_graph_rect,
                self.network.W_AB,
                self.network.W_BA,
                "Connectivity Graph (top 10%)"
            )

            self.draw_rate_plots(self.rate_panel)

            # Draw UI on top
            self.ui_manager.draw_ui(self.screen)

            pygame.display.flip()

        pygame.quit()
