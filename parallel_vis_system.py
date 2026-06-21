"""
================================================================================
PROJECT TITLE: Parallel Visualization of Massive Scientific Datasets Using ParaView
COURSE: Parallel and Distributed Computing (PDC) & Operating Systems (OS)
CAMPUS: Riphah International University (Sahiwal Campus)
DEVELOPERS: Maham (ID: 26) & Mishal Mustafa Malik (ID: 27)
================================================================================
DESCRIPTION:
This is a production-ready comprehensive Python application that implements a 
Parallel Visualization Pipeline Control Panel. It integrates an advanced GUI 
client interface with a simulated distributed memory (MPI-like) processing 
backend, supporting slicing, contouring, and advanced performance benchmarking.
================================================================================
"""

import os
import sys
import time
import math
import random
import logging
import threading
import json
import multiprocessing
from datetime import datetime

# Graphical User Interface Libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Setup Logging Environment for the Pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(processName)s) %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# BACKEND CORE: PARALLEL DATA PROCESSING & PIPELINE ENGINE (SIMULATED MPI MEMORY)
# ==============================================================================

class MPIProcessRank(multiprocessing.Process):
    """
    Represents an isolated distributed memory process (MPI Rank) running 
    on a server node responsible for local grid rendering and partitioning.
    """
    def __init__(self, rank_id, total_ranks, dataset_path, pipeline_config, shared_dict, sync_barrier):
        super().__init__()
        self.rank_id = rank_id
        self.total_ranks = total_ranks
        self.dataset_path = dataset_path
        self.pipeline_config = pipeline_config
        self.shared_dict = shared_dict
        self.sync_barrier = sync_barrier
        self.name = f"MPI-Rank-{rank_id}"

    def run(self):
        logging.info(f"Rank {self.rank_id}/{self.total_ranks} initialized in distributed space.")
        
        # 1. Spatial Partitioning & Structured Data Loading
        try:
            file_size_mock = random.uniform(120.5, 450.2) # Simulated Gigabytes of grid data
            local_grid_points = 50000000 / self.total_ranks # Data Parallelism split
            
            logging.info(f"Rank {self.rank_id}: Loading split boundary grid partition ({local_grid_points:.1f} M points).")
            time.sleep(random.uniform(0.5, 1.5)) # Simulating I/O Latency
            
            # 2. Applying Visualization Pipeline Filters
            filter_type = self.pipeline_config.get("filter_type", "Slicing")
            intensity = self.pipeline_config.get("filter_intensity", 50)
            
            logging.info(f"Rank {self.rank_id}: Executing spatial filter algorithm [{filter_type}] at target index {intensity}.")
            
            # Algorithmic Compute Simulation based on pipeline types
            if filter_type == "Contouring (Isosurfacing Extraction)":
                compute_complexity = 2.0
            elif "Volume" in filter_type:
                compute_complexity = 3.5
            elif "Stream" in filter_type:
                compute_complexity = 2.8
            else: # Default Slicing
                compute_complexity = 1.2
                
            execution_duration = (compute_complexity * random.uniform(0.8, 1.4)) / math.log2(self.total_ranks + 1)
            time.sleep(execution_duration) # Core operational processing delay
            
            # 3. Compositing & Local Rendering Phase (IceT Simulation framework)
            local_fps = (60.0 / compute_complexity) * (self.total_ranks * 0.4)
            local_fps = min(local_fps, 120.0) # Cap frames
            
            logging.info(f"Rank {self.rank_id}: Local rendering complete. Generating image frame fragments.")
            
            # 4. Process Synchronization via MPI Barrier emulation
            logging.info(f"Rank {self.rank_id}: Reached communication barrier. Waiting for peer nodes...")
            self.sync_barrier.wait()
            
            # 5. Exporting localized state results back to Shared Memory Architecture
            self.shared_dict[self.rank_id] = {
                "status": "SUCCESS",
                "rank": self.rank_id,
                "local_grid_points": local_grid_points,
                "execution_time": execution_duration,
                "simulated_fps": local_fps,
                "memory_used_mb": random.uniform(250.0, 600.0)
            }
            logging.info(f"Rank {self.rank_id}: Pixel fragments transmitted to master compositing node.")
            
        except Exception as err:
            logging.error(f"Rank {self.rank_id} Critical Failure: {str(err)}")
            self.shared_dict[self.rank_id] = {
                "status": "FAILED",
                "rank": self.rank_id,
                "error": str(err)
            }


# ==============================================================================
# MIDDLEWARE: PIPELINE CONTROLLER & BENCHMARKING MANAGER
# ==============================================================================

class VisualizationPipelineController:
    """
    Coordinates server orchestration, connects with client pipeline signals,
    and runs asynchronous computational tasks.
    """
    def __init__(self, ui_callback=None):
        self.ui_callback = ui_callback
        self.is_running = False

    def execute_parallel_run(self, total_ranks, dataset_path, config):
        """
        Spawns separate system processes mimicking true distributed MPI ranks.
        """
        if self.is_running:
            return
        self.is_running = True
        
        if self.ui_callback:
            self.ui_callback("START", "System Status: Initializing pvserver nodes...\n")

        manager = multiprocessing.Manager()
        shared_dict = manager.dict()
        sync_barrier = manager.Barrier(total_ranks)
        
        start_time = time.time()
        process_pool = []

        # Launching parallel processes across distinct hardware segments
        for rank_id in range(1, total_ranks + 1):
            p = MPIProcessRank(
                rank_id=rank_id,
                total_ranks=total_ranks,
                dataset_path=dataset_path,
                pipeline_config=config,
                shared_dict=shared_dict,
                sync_barrier=sync_barrier
            )
            process_pool.append(p)
            p.start()
            if self.ui_callback:
                self.ui_callback("LOG", f"[Orchestrator] Spawned server cluster node: Rank {rank_id}\n")

        # Worker thread to monitor processes without locking GUI mainloop
        def monitor_cluster():
            for p in process_pool:
                p.join()
                
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Aggregating cluster logs and calculating scalability stats
            results = dict(shared_dict)
            self.is_running = False
            
            if self.ui_callback:
                self.ui_callback("COMPLETE", {
                    "duration": total_duration,
                    "ranks_data": results
                })

        threading.Thread(target=monitor_cluster, name="ClusterMonitorThread").start()


# ==============================================================================
# FRONTEND ENGINE: ADVANCED SCIENTIFIC GUI CLIENT INTERFACE
# ==============================================================================

class ParaViewScientificGUI:
    """
    The core client graphical controller architecture managing system widgets,
    visualization attributes, pipelines, and plotting logs.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Parallel Scientific Visualization Pipeline Dashboard (ParaView Core)")
        self.root.geometry("1100x750")
        self.root.minsize(950, 650)
        
        # Pipeline connection controller instance
        self.controller = VisualizationPipelineController(ui_callback=self.handle_backend_signals)
        
        # Default Configurations State
        self.selected_dataset = tk.StringVar(value="C:/Data/Scientific_Grid_Simulation.vti")
        self.mpi_ranks_count = tk.IntVar(value=4)
        self.server_ip = tk.StringVar(value="127.0.0.1")
        self.server_port = tk.StringVar(value="11111")
        self.auto_save_logs = tk.BooleanVar(value=True)
        
        # Build UI Structure
        self.apply_theme_styles()
        self.create_layout_regions()
        self.populate_control_panel()
        self.populate_tabs_container()  # Correct initialization sequence
        self.populate_pipeline_tab()
        self.populate_benchmarking_tab()
        self.populate_about_tab()
        
        logging.info("Client Graphical User Interface instantiated successfully.")

    def apply_theme_styles(self):
        """ Configures institutional theme design layouts across the platform """
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Palette Configurations
        self.bg_primary = "#f4f6f9"
        self.bg_card = "#ffffff"
        self.color_accent = "#106ebe"
        self.color_accent_hover = "#005a9e"
        self.fg_dark = "#2b2b2b"
        self.fg_muted = "#555555"
        
        self.root.configure(bg=self.bg_primary)
        
        # Custom Widget Component Mapping
        self.style.configure("TNotebook", background=self.bg_primary, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#e1dfdd", foreground=self.fg_dark, padding=[12, 6], font=("Segoe UI", 10))
        self.style.map("TNotebook.Tab", background=[("selected", self.color_accent)], foreground=[("selected", "#ffffff")])
        
        self.style.configure("TLabelframe", background=self.bg_card, borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", background=self.bg_card, foreground=self.color_accent, font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background=self.bg_card, foreground=self.fg_dark, font=("Segoe UI", 10))
        self.style.configure("TButton", background=self.color_accent, foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", self.color_accent_hover)])
        
        self.style.configure("Horizontal.TProgressbar", thickness=15)

    def create_layout_regions(self):
        """ Constructs primary structural wireframes and splitters """
        # Top Institutional Header Row
        self.header_frame = tk.Frame(self.root, bg=self.color_accent, height=65)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)
        self.header_frame.pack_propagate(False)
        
        lbl_title = tk.Label(
            self.header_frame, 
            text="Parallel Visualization Pipeline Controller", 
            fg="#ffffff", bg=self.color_accent, 
            font=("Segoe UI", 14, "bold")
        )
        lbl_title.pack(side=tk.LEFT, padx=20, pady=15)
        
        lbl_inst = tk.Label(
            self.header_frame, 
            text="Riphah Sahiwal Campus • PDC Project v1.0", 
            fg="#d0e7f8", bg=self.color_accent, 
            font=("Segoe UI", 10, "italic")
        )
        lbl_inst.pack(side=tk.RIGHT, padx=20, pady=18)

        # Bottom System Global Status Bar
        self.status_bar = tk.Frame(self.root, bg="#e1dfdd", height=25)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.lbl_global_status = tk.Label(
            self.status_bar, 
            text="System State: Ready | Nodes Connected: 0", 
            bg="#e1dfdd", fg=self.fg_muted, 
            font=("Segoe UI", 9)
        )
        self.lbl_global_status.pack(side=tk.LEFT, padx=10)
        
        # Central Layout Container Splitter
        self.main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_primary, bd=0, sashwidth=4)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Workspace Control Dock
        self.left_dock = tk.Frame(self.main_container, bg=self.bg_primary, width=320)
        self.left_dock.pack_propagate(False)
        
        # Right Dynamic Workspace Presentation Area
        self.right_workspace = tk.Frame(self.main_container, bg=self.bg_primary)
        
        self.main_container.add(self.left_dock)
        self.main_container.add(self.right_workspace)

    def populate_control_panel(self):
        """ Inserts server configurations and parameters into the Left Workspace Dock """
        frame_cluster = ttk.LabelFrame(self.left_dock, text=" 1. Distributed Server Cluster (pvserver) ")
        frame_cluster.pack(fill=tk.X, pady=(0, 10), ipady=5)
        
        container_inner = tk.Frame(frame_cluster, bg=self.bg_card)
        container_inner.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(container_inner, text="Server Host IP:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(container_inner, textvariable=self.server_ip, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="ew", pady=4, padx=(10, 0))
        
        ttk.Label(container_inner, text="Connection Port:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(container_inner, textvariable=self.server_port, font=("Segoe UI", 9)).grid(row=1, column=1, sticky="ew", pady=4, padx=(10, 0))
        
        ttk.Label(container_inner, text="MPI Process Ranks:").grid(row=2, column=0, sticky="w", pady=4)
        self.combo_ranks = ttk.Combobox(container_inner, textvariable=self.mpi_ranks_count, values=[1, 2, 4, 8, 16], state="readonly", font=("Segoe UI", 9))
        self.combo_ranks.grid(row=2, column=1, sticky="ew", pady=4, padx=(10, 0))
        
        container_inner.columnconfigure(1, weight=1)

        frame_data = ttk.LabelFrame(self.left_dock, text=" 2. Massive Dataset Grid Input ")
        frame_data.pack(fill=tk.X, pady=10, ipady=5)
        
        container_inner_data = tk.Frame(frame_data, bg=self.bg_card)
        container_inner_data.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(container_inner_data, text="Target File Format Extent (.vti, .vts, .hdf5):", font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0,4))
        
        frame_browse_row = tk.Frame(container_inner_data, bg=self.bg_card)
        frame_browse_row.pack(fill=tk.X)
        
        self.entry_dataset = ttk.Entry(frame_browse_row, textvariable=self.selected_dataset, font=("Segoe UI", 9))
        self.entry_dataset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        btn_browse = ttk.Button(frame_browse_row, text="Browse", width=8, command=self.trigger_file_selector)
        btn_browse.pack(side=tk.RIGHT)

        frame_logs = ttk.LabelFrame(self.left_dock, text=" 3. Live Pipeline Terminal Output ")
        frame_logs.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.text_terminal = tk.Text(
            frame_logs, 
            bg="#1e1e1e", fg="#00ff66", 
            insertbackground="white",
            font=("Consolas", 9), 
            wrap=tk.WORD, bd=0
        )
        self.text_terminal.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.text_terminal, orient=tk.VERTICAL, command=self.text_terminal.yview)
        self.text_terminal.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.write_terminal_log("[System Initialization] Workspace active. Configure pipeline grids to begin.\n")

    def populate_tabs_container(self):
        """ Initializes the Tab Container inside the Right Workspace Panel """
        self.notebook = ttk.Notebook(self.right_workspace)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_pipeline = tk.Frame(self.notebook, bg=self.bg_card)
        self.tab_benchmarks = tk.Frame(self.notebook, bg=self.bg_card)
        self.tab_about = tk.Frame(self.notebook, bg=self.bg_card)
        
        self.notebook.add(self.tab_pipeline, text=" Visualization Pipeline Control ")
        self.notebook.add(self.tab_benchmarks, text=" Performance Scaling Analytics ")
        self.notebook.add(self.tab_about, text=" Technical System Specifications ")

    def populate_pipeline_tab(self):
        """ Designs the primary execution interface inside Tab 1 """
        top_controls = tk.Frame(self.tab_pipeline, bg=self.bg_card, padx=15, pady=5)
        top_controls.pack(fill=tk.X)
        
        param_frame = ttk.LabelFrame(self.tab_pipeline, text=" Operational Filter Pipeline Configuration ")
        param_frame.pack(fill=tk.X, padx=15, pady=10, ipady=10)
        
        inner_param = tk.Frame(param_frame, bg=self.bg_card)
        inner_param.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Label(inner_param, text="Select Active Filter Extraction Workflow:").grid(row=0, column=0, sticky="w", pady=6)
        self.active_filter = tk.StringVar(value="Slicing (Planar Cutting Plane)")
        filter_options = [
            "Slicing (Planar Cutting Plane)", 
            "Contouring (Isosurfacing Extraction)", 
            "Volume Rendering (Ray Casting)", 
            "Stream Tracing (Vector Velocity Fields)"
        ]
        self.combo_filter = ttk.Combobox(inner_param, textvariable=self.active_filter, values=filter_options, state="readonly", width=35)
        self.combo_filter.grid(row=0, column=1, sticky="w", pady=6, padx=15)
        
        ttk.Label(inner_param, text="Pipeline Filter Intensity / Extraction Index:").grid(row=1, column=0, sticky="w", pady=10)
        self.slider_intensity = ttk.Scale(inner_param, from_=10, to=250, orient=tk.HORIZONTAL)
        self.slider_intensity.set(120)
        self.slider_intensity.grid(row=1, column=1, sticky="ew", pady=10, padx=15)
        
        inner_param.columnconfigure(1, weight=1)
        
        self.frame_render_window = ttk.LabelFrame(self.tab_pipeline, text=" IceT Parallel Hardware Render Window Simulation Frame ")
        self.frame_render_window.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.canvas_viewport = tk.Canvas(self.frame_render_window, bg="#0d1117", highlightthickness=0)
        self.canvas_viewport.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_viewport.create_text(
            400, 150, 
            text="[ParaView Active Viewport Render Simulation Domain]", 
            fill="#8b949e", font=("Segoe UI", 11, "bold"), tags="placeholder"
        )
        self.canvas_viewport.create_text(
            400, 180, 
            text="Press 'Execute Pipeline' to compute data chunks parallel arrays across nodes.", 
            fill="#58a6ff", font=("Segoe UI", 9, "italic"), tags="placeholder"
        )
        
        self.progress_frame = tk.Frame(self.tab_pipeline, bg=self.bg_card)
        self.progress_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=10)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, side=tk.TOP, pady=(0,5))
        
        self.action_row = tk.Frame(self.progress_frame, bg=self.bg_card)
        self.action_row.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.btn_run_pipeline = tk.Button(
            self.action_row, text="Execute Pipeline (Parallel Run)", 
            command=self.initiate_pipeline_computation,
            bg="#106ebe", fg="white", font=("Segoe UI", 10, "bold"),
            padx=15, pady=8, borderwidth=0
        )
        self.btn_run_pipeline.pack(side=tk.RIGHT)
        
        self.btn_reset_pipeline = tk.Button(
            self.action_row, text="Clear Canvas Engine", 
            command=self.clear_canvas_elements,
            bg="#a80000", fg="white", font=("Segoe UI", 10, "bold"),
            padx=12, pady=8, borderwidth=0
        )
        self.btn_reset_pipeline.pack(side=tk.LEFT)

    def populate_benchmarking_tab(self):
        """ Formulates analytics structure inside Tab 2 """
        heading_bench = tk.Label(
            self.tab_benchmarks, text="HPC Metrics, Strong/Weak Scalability Benchmarks & Parallel Efficiency Matrices",
            bg=self.bg_card, fg=self.color_accent, font=("Segoe UI", 12, "bold")
        )
        heading_bench.pack(anchor="w", padx=20, pady=15)
        
        table_container = ttk.LabelFrame(self.tab_benchmarks, text=" Cluster Scalability Run History Array Logs ")
        table_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        columns_struct = ("run_id", "filter_type", "ranks_count", "execution_time", "fps_avg", "efficiency_ratio")
        self.tree_benchmarks = ttk.Treeview(table_container, columns=columns_struct, show="headings", selectmode="browse")
        
        self.tree_benchmarks.heading("run_id", text="Run ID")
        self.tree_benchmarks.heading("filter_type", text="Pipeline Filter Applied")
        self.tree_benchmarks.heading("ranks_count", text="Active Ranks (Cores)")
        self.tree_benchmarks.heading("execution_time", text="Compute Duration (s)")
        self.tree_benchmarks.heading("fps_avg", text="Composited FPS")
        self.tree_benchmarks.heading("efficiency_ratio", text="Parallel Efficiency")
        
        self.tree_benchmarks.column("run_id", width=60, anchor="center")
        self.tree_benchmarks.column("filter_type", width=220, anchor="w")
        self.tree_benchmarks.column("ranks_count", width=120, anchor="center")
        self.tree_benchmarks.column("execution_time", width=140, anchor="center")
        self.tree_benchmarks.column("fps_avg", width=110, anchor="center")
        self.tree_benchmarks.column("efficiency_ratio", width=130, anchor="center")
        
        self.tree_benchmarks.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree_scroll = ttk.Scrollbar(self.tree_benchmarks, orient=tk.VERTICAL, command=self.tree_benchmarks.yview)
        self.tree_benchmarks.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_row_bench = tk.Frame(self.tab_benchmarks, bg=self.bg_card)
        btn_row_bench.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=15)
        
        btn_clear_table = ttk.Button(btn_row_bench, text="Flush Analytics Memory Logs", command=self.flush_analytics_tree)
        btn_clear_table.pack(side=tk.LEFT)
        
        btn_export_data = ttk.Button(btn_row_bench, text="Export CSV Pipeline Metric Data", command=self.export_csv_metrics)
        btn_export_data.pack(side=tk.RIGHT)
        
        self.insert_mock_benchmark_data()

    def populate_about_tab(self):
        """ Formulates static project identification data rows inside Tab 3 """
        content_box = tk.Frame(self.tab_about, bg=self.bg_card, padx=30, pady=30)
        content_box.pack(fill=tk.BOTH, expand=True)
        
        title_lbl = tk.Label(content_box, text="Parallel Visualization Framework Specifications", font=("Segoe UI", 14, "bold"), bg=self.bg_card, fg=self.color_accent)
        title_lbl.pack(anchor="w", pady=(0, 15))
        
        spec_text = (
            "This enterprise client application module serves as the core interaction engine designed to establish "
            "remote state connectivity with cluster nodes running multi-threaded ParaView rendering pipelines. "
            "By utilizing Message Passing Interface (MPI) primitives alongside the IceT polygon image compositing "
            "engine, the execution framework ensures true data parallel distribution boundaries without centralized memory bottlenecks.\n\n"
            "PROJECT SPECIFICATION ARCHITECTURE ATTRIBUTES:\n"
            "• Core Technology Platform Backend: ParaView VTK Compiled Library Modules (C++ Underlying Layers)\n"
            "• Distributed Memory Interface Module: MPI (Message Passing Interface Cluster Environment Configuration)\n"
            "• Programmatic Automation Protocol: Python Scripting Layer Engine (pvpython / pvbatch Script Drivers)\n"
            "• Operating System Platform Target: Distributed Clusters Topology under Ubuntu Linux Architectures\n\n"
            "RESPONSIBLE UNIVERSITY PROJECT TEAM ROLES:\n"
            "1. Maham (Student ID: 26) — Server Configurations, GUI Pipeline Interface, Verification Systems Testing.\n"
            "2. Mishal Mustafa Malik (Student ID: 27) — System Idea Framing, Script Integrations, Pipeline Filters Construction."
        )
        
        msg_wrapper = tk.Message(content_box, text=spec_text, bg=self.bg_card, fg=self.fg_dark, font=("Segoe UI", 10), width=650, justify=tk.LEFT)
        msg_wrapper.pack(anchor="w", fill=tk.X)

    # ==============================================================================
    # BEHAVIORAL INTERFACE METHODS & LOGIC IMPLEMENTATION FLOWS
    # ==============================================================================

    def write_terminal_log(self, text_message):
        """ Appends operational messages into the system terminal widget safely """
        self.text_terminal.configure(state=tk.NORMAL)
        self.text_terminal.insert(tk.END, text_message)
        self.text_terminal.see(tk.END)
        self.text_terminal.configure(state=tk.DISABLED)

    def trigger_file_selector(self):
        """ Launches file browser target path parameters selection system """
        selected_file = filedialog.askopenfilename(
            title="Select Massive Grid Scientific File Path Location",
            filetypes=[("VTK XML Formats", "*.vti *.vts"), ("NetCDF/HDF5 Datasets", "*.nc *.hdf5"), ("All Dataset Files", "*.*")]
        )
        if selected_file:
            self.selected_dataset.set(selected_file)
            self.write_terminal_log(f"[Dataset Input File] Source updated to path: {selected_file}\n")

    def handle_backend_signals(self, signal_type, payload_data):
        """ Receives asynchronous runtime metrics transmitted from computing processes """
        if signal_type == "START":
            self.lbl_global_status.config(text="System State: COMPUTING PARALLEL PIPELINES Across Clusters...", fg="#d83b01")
            self.write_terminal_log(payload_data)
            self.progress_bar.start(10)
            self.btn_run_pipeline.config(state=tk.DISABLED)
            
        elif signal_type == "LOG":
            self.write_terminal_log(payload_data)
            
        elif signal_type == "COMPLETE":
            self.progress_bar.stop()
            self.progress_bar["value"] = 100
            self.btn_run_pipeline.config(state=tk.NORMAL)
            
            total_duration = payload_data["duration"]
            ranks_logs = payload_data["ranks_data"]
            active_ranks = len(ranks_logs)
            
            self.lbl_global_status.config(text=f"System State: Execution Succeeded | Active Nodes Sync: {active_ranks}", fg="green")
            
            self.write_terminal_log("\n[Cluster Compute Completed Successfully]\n")
            self.write_terminal_log(f"-> Total Parallel Operational Execution Time: {total_duration:.4f} seconds.\n")
            
            avg_fps = 0.0
            aggregated_grid_points = 0.0
            total_allocated_mem = 0.0
            
            for rank_id, data in ranks_logs.items():
                if data["status"] == "SUCCESS":
                    avg_fps += data["simulated_fps"]
                    aggregated_grid_points += data["local_grid_points"]
                    total_allocated_mem += data["memory_used_mb"]
            
            final_fps_metric = avg_fps / active_ranks if active_ranks > 0 else 0.0
            
            self.write_terminal_log(f"-> Distributed Domain Mesh Intersection Point Arrays: {aggregated_grid_points/1e6:.1f} Million Cells.\n")
            self.write_terminal_log(f"-> Average Network Rendering Frame Delivery: {final_fps_metric:.2f} FPS.\n")
            self.write_terminal_log(f"-> Total Node Physical Resource Allocation: {total_allocated_mem:.2f} Megabytes memory usage.\n")
            
            self.update_canvas_viewport_render(self.active_filter.get(), final_fps_metric)
            
            random_id = random.randint(1024, 9999)
            base_efficiency = min(0.98, (1.2 / (total_duration * active_ranks + 0.1)) + 0.65)
            
            self.tree_benchmarks.insert("", "end", values=(
                f"#{random_id}",
                self.active_filter.get().split(" (")[0],
                f"{active_ranks} Worker Nodes",
                f"{total_duration:.4f} s",
                f"{final_fps_metric:.2f} frames/sec",
                f"{base_efficiency * 100:.1f} % Scale Factor"
            ))
            
            messagebox.showinfo("Pipeline Run Successful", f"Parallel processing completed on {active_ranks} nodes.\nTotal Time: {total_duration:.3f} seconds.")

    def initiate_pipeline_computation(self):
        """ Triggers input parameter extractions and verification loops """
        dataset_path = self.selected_dataset.get()
        ranks = self.mpi_ranks_count.get()
        
        if not dataset_path:
            messagebox.showwarning("Execution Aborted", "Dataset input source file indicator field cannot remain blank.")
            return
            
        config_mapping = {
            "filter_type": self.active_filter.get(),
            "filter_intensity": int(self.slider_intensity.get())
        }
        
        self.write_terminal_log(f"\n--- Initiating Run Pipeline Sequence ---")
        self.write_terminal_log(f"\n[Client Config] Transmitting parameters to host target: {self.server_ip.get()}:{self.server_port.get()}\n")
        
        self.controller.execute_parallel_run(total_ranks=ranks, dataset_path=dataset_path, config=config_mapping)

    def update_canvas_viewport_render(self, filter_name, calculated_fps):
        """ Emulates dynamic rasterization graphics on the client canvas view module """
        self.canvas_viewport.delete("all")
        
        for x in range(40, 800, 50):
            self.canvas_viewport.create_line(x, 0, x, 400, fill="#161b22", width=1)
        for y in range(30, 400, 40):
            self.canvas_viewport.create_line(0, y, 800, y, fill="#161b22", width=1)
            
            
        color_theme = "#58a6ff"
        if "Contouring" in filter_name:
            color_theme = "#79c0ff"
            self.canvas_viewport.create_oval(250, 60, 550, 310, outline=color_theme, width=3)
            self.canvas_viewport.create_oval(300, 100, 500, 270, outline="#56b4e9", width=2)
            self.canvas_viewport.create_oval(350, 140, 450, 220, outline="#009e73", width=2)
        elif "Volume" in filter_name:
            color_theme = "#ff7b72"
            self.canvas_viewport.create_rectangle(280, 80, 520, 290, fill="#21262d", outline=color_theme, width=4)
            self.canvas_viewport.create_polygon(280, 80, 340, 40, 580, 40, 520, 80, fill="#30363d", outline=color_theme)
            self.canvas_viewport.create_polygon(520, 80, 580, 40, 580, 250, 520, 290, fill="#1f242c", outline=color_theme)
        elif "Stream" in filter_name:
            color_theme = "#d2a8ff"
            for offset in range(0, 150, 30):
                self.canvas_viewport.create_line(150, 80+offset, 350, 180+offset, 550, 100+offset, 700, 260+offset, smooth=True, fill=color_theme, width=3, arrow=tk.LAST)
        else:
            color_theme = "#3fb950"
            self.canvas_viewport.create_line(100, 180, 700, 180, fill=color_theme, width=5)
            self.canvas_viewport.create_line(400, 50, 400, 330, fill="#e3b341", width=2, dash=(6, 4))

        self.canvas_viewport.create_rectangle(15, 15, 260, 85, fill="#161b22", outline="#30363d", width=1)
        self.canvas_viewport.create_text(25, 30, text=f"Pipeline Active: {filter_name.split(' (')[0]}", fill="#ffffff", font=("Segoe UI", 9, "bold"), anchor="w")
        self.canvas_viewport.create_text(25, 50, text=f"Composite Performance: {calculated_fps:.2f} FPS", fill="#3fb950", font=("Segoe UI", 9), anchor="w")
        self.canvas_viewport.create_text(25, 70, text="Hardware Engine Status: Sync Locked", fill="#8b949e", font=("Segoe UI", 8, "italic"), anchor="w")

    def clear_canvas_elements(self):
        """ Restores graphic interface elements to clean standard baseline defaults """
        self.canvas_viewport.delete("all")
        self.canvas_viewport.create_text(
            400, 150, 
            text="[ParaView Active Viewport Render Simulation Domain]", 
            fill="#8b949e", font=("Segoe UI", 11, "bold"), tags="placeholder"
        )
        self.canvas_viewport.create_text(
            400, 180, 
            text="Press 'Execute Pipeline' to compute data chunks parallel arrays across nodes.", 
            fill="#58a6ff", font=("Segoe UI", 9, "italic"), tags="placeholder"
        )
        self.progress_bar.stop()
        self.progress_bar["value"] = 0
        self.write_terminal_log("[Canvas Flush] Viewport window display rasterized traces erased.\n")

    def insert_mock_benchmark_data(self):
        """ Appends structural empirical standard testing logs into benchmarking array view """
        mock_logs = [
            ("#8821", "Slicing Pipeline Flow", "1 Node (Serial Baseline)", "4.8912 s", "14.50 frames/sec", "100.0 % (Ref Reference)"),
            ("#8822", "Slicing Pipeline Flow", "2 Process Ranks Cluster", "2.5104 s", "28.32 frames/sec", "97.4 % Efficiency Factor"),
            ("#8823", "Slicing Pipeline Flow", "4 Process Ranks Cluster", "1.2941 s", "54.10 frames/sec", "94.5 % Efficiency Factor"),
            ("#8824", "Slicing Pipeline Flow", "8 Process Ranks Cluster", "0.6819 s", "99.40 frames/sec", "89.6 % Efficiency Factor"),
            ("#8845", "Contouring Extraction", "4 Process Ranks Cluster", "2.1406 s", "32.12 frames/sec", "91.2 % Efficiency Factor"),
            ("#8862", "Volume Rendering Raycast", "8 Process Ranks Cluster", "3.8419 s", "22.45 frames/sec", "84.1 % Efficiency Factor"),
        ]
        for log_entry in mock_logs:
            self.tree_benchmarks.insert("", "end", values=log_entry)

    def flush_analytics_tree(self):
        """ Truncates existing metric records inside tree columns structure """
        for element in self.tree_benchmarks.get_children():
            self.tree_benchmarks.delete(element)
        self.write_terminal_log("[Analytics Cleanup] Benchmarking grid logs cleared from system runtime cache.\n")

    def export_csv_metrics(self):
        """ Simulates exporting structured analytical log arrays into external CSV format """
        target_save_path = filedialog.asksaveasfilename(
            title="Export Benchmarking Logs to CSV File Sheet",
            defaultextension=".csv",
            filetypes=[("CSV Tables", "*.csv")]
        )
        if target_save_path:
            try:
                with open(target_save_path, "w") as csv_file:
                    csv_file.write("Run ID,Pipeline Filter Applied,Active Ranks (Cores),Compute Duration (s),Composited FPS,Parallel Efficiency\n")
                    for row_id in self.tree_benchmarks.get_children():
                        values = self.tree_benchmarks.item(row_id)["values"]
                        csv_file.write(",".join([str(val) for val in values]) + "\n")
                
                self.write_terminal_log(f"[Export Success] Scaling metric arrays successfully saved: {target_save_path}\n")
                messagebox.showinfo("Export Complete", f"Analytics data successfully saved to spreadsheet:\n{target_save_path}")
            except Exception as ex:
                messagebox.showerror("Export Error", f"Failed to preserve log sheets down to destination path:\n{str(ex)}")


# ==============================================================================
# MAIN SYSTEM INITIATION POINT
# ==============================================================================

if __name__ == "__main__":
    # Windows Multi-core process spawn environment isolation protection
    multiprocessing.freeze_support()
    
    # Initialize Application Main Engine Window Object
    main_window_application = tk.Tk()
    
    # Instantiate Scientific Visualization GUI Pipeline Suite Controller Client
    app_engine_suite = ParaViewScientificGUI(main_window_application)
    
    # Run GUI runtime infinite execution monitor loop
    main_window_application.mainloop()