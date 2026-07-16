import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import csv
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator

def run_benchmark(model_f, labels, in_vid, cpu_thread=3, probability=0.5):
    inference_results = []
    entry = {}
    for model_path in sorted(model_f):
        model_name = model_path.name
        output_name = f"{model_name}_{cpu_thread}_{probability}_out_vid.mp4"
        process = subprocess.run(
            [
                "python3",
                "run_vd.py",
                str(model_path),
                labels,
                in_vid,
                output_name,
                str(cpu_thread),
                str(probability),
                "NOGEN",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        inference = float(process.stdout.strip())
        if "edgetpu" not in model_name:
            common_model = "_".join(model_name.split("_")[:-2])
            entry = {
                "Model": common_model,
                "CPU": inference,
            }
        else:
            entry["EdgeTPU"] = inference
            inference_results.append(entry)
    return inference_results

if __name__ == "__main__":
    # Iterates through the folder and filters out subdirectories
    folder_path = Path("model")
    model_files = [p for p in folder_path.iterdir() if p.is_file()]
    headers = ["Model", "CPU", "EdgeTPU"]
    # Output data
    benchmark_data = []
    # ========================== Benchmark ==========================
    tests = [(1, 0.5), (2, 0.5), (3, 0.5)]
    for cpu_thread, probability in tests:
        print(f"=== Start test cpu: {cpu_thread} probability: {probability} ===")
        intf_results = run_benchmark(model_files, "coco_labels.txt", "in_vid.mp4", cpu_thread, probability)
        benchmark_data.append(intf_results.copy())
        print("========================= Finished ============================\n")
    print("Benchmark data", benchmark_data)
    print("========== Draw plots ==========")
    # ========================== Output ==========================
    models = [d['Model'] for d in benchmark_data[0]]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))

    for thread, group in enumerate(benchmark_data, start=1):

        cpu = [d['CPU'] for d in group]
        tpu = [d['EdgeTPU'] for d in group]

        ax.plot(
            x,
            cpu,
            marker='o',
            linewidth=2,
            label=f'CPU ({thread} thread{"s" if thread > 1 else ""})'
        )

        ax.plot(
            x,
            tpu,
            marker='s',
            linewidth=2,
            linestyle='--',
            label=f'Edge TPU (CPU {thread} thread{"s" if thread > 1 else ""})'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_ylabel("Average Inference Time (ms)")
    ax.set_title("CPU vs Edge TPU Benchmark")

    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.25)

    ax.legend()

    plt.tight_layout()
    plt.savefig("comparison_plot_perf.png", dpi=150)
    print("Script done!")
