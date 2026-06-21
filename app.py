import tkinter as tk
from tkinter import ttk, messagebox
import multiprocessing
import time

# --- BACKEND: Parallel Processing Function ---
# Yeh function simulate karega ke backend par massive data chunks parallel process ho rahe hain
def process_data_chunk(chunk_id, return_dict):
    print(f"[Backend] Process {chunk_id} started processing its data grid.")
    
    # Complex visualization/rendering simulation
    time.sleep(3) 
    
    # Result return karna
    return_dict[chunk_id] = f"Chunk {chunk_id} Rendered Successfully ✔"
    print(f"[Backend] Process {chunk_id} finished.")

# --- FRONTEND: Tkinter GUI Class ---
class ParaViewParallelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parallel Visualization Controller (Dev Mode)")
        self.root.geometry("500x400")
        self.root.configure(bg="#f0f0f0")

        # Title Label
        title_label = tk.Label(root, text="PDC Scientific Visualization Pipeline", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=15)

        # Frame for controls
        control_frame = tk.Frame(root, bg="#f0f0f0")
        control_frame.pack(pady=10)

        # Dropdown for Core/Process count
        tk.Label(control_frame, text="Select Parallel Processes (MPI Ranks Simulation):", bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5)
        self.process_count = tk.IntVar(value=2)
        self.process_dropdown = ttk.Combobox(control_frame, textvariable=self.process_count, values=[2, 4, 8], width=5, state="readonly")
        self.process_dropdown.grid(row=0, column=1, padx=5, pady=5)

        # Pipeline Operation Buttons
        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=15)

        self.run_btn = tk.Button(btn_frame, text="Run Parallel Pipeline", command=self.start_parallel_pipeline, bg="#0078d4", fg="white", font=("Arial", 10, "bold"), padx=10, pady=5)
        self.run_btn.grid(row=0, column=0, padx=10)

        # Progress / Status Section
        self.status_label = tk.Label(root, text="System Status: Ready", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="green")
        self.status_label.pack(pady=10)

        # Output Terminal Box in GUI
        tk.Label(root, text="Console Output:", bg="#f0f0f0", font=("Arial", 9, "bold")).pack(anchor="w", padx=25)
        self.output_box = tk.Text(root, height=8, width=55, bg="#222", fg="#00ff00", font=("Consolas", 9))
        self.output_box.pack(pady=5, padx=25)

    def log_message(self, message):
        self.output_box.insert(tk.END, message + "\n")
        self.output_box.see(tk.END)

    def start_parallel_pipeline(self):
        num_processes = self.process_count.get()
        self.status_label.config(text="Status: Processing Parallel Grids...", fg="orange")
        self.log_message(f"\n[Client] Initializing execution with {num_processes} concurrent workers...")
        self.root.update()

        # Multiprocessing setup (Simulating MPI distributed memory behavior)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        start_time = time.time()

        # Launching parallel processes
        for i in range(num_processes):
            p = multiprocessing.Process(target=process_data_chunk, args=(i+1, return_dict))
            processes.append(p)
            p.start()
            self.log_message(f"[Client] Process Rank {i+1} spawned.")

        # Waiting for all processes to sync (Simulating MPI Barrier)
        for p in processes:
            p.join()

        end_time = time.time()

        # Fetching results from dictionary
        self.log_message("\n--- Pipeline Execution Summary ---")
        for rank, result in return_dict.items():
            self.log_message(f"Rank {rank}: {result}")

        self.log_message(f"Total Execution Time: {end_time - start_time:.2f} seconds")
        self.status_label.config(text="Status: Execution Completed Successfully", fg="green")
        messagebox.showinfo("Success", f"Parallel rendering done on {num_processes} cores!")

# --- Execution Block ---
if __name__ == '__main__':
    # Windows par multiprocessing safe chalane ke liye freeze_support zaroori hai
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = ParaViewParallelGUI(root)
    root.mainloop()