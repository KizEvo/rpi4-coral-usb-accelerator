import subprocess
import sys
from pathlib import Path
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

def run_benchmark(model_f, labels, in_vid, cpu_thread=3, probability=0.5):
    inference_results = []
    dict = {}
    # Sort so that CPU version of a model is ran first, then Edge TPU version.
    sorted_model_files = sorted(model_f, key=str)
    for model_posixpath in sorted_model_files:
        # Preprocess the model name.
        model_str_path = str(model_posixpath)
        model_name = model_str_path.split("/")[-1]
        print(f"Running model: ${model_name}")
        # Runs script.py and waits for it to finish.
        output_name = f"{model_name}_{cpu_thread}_{probability}_out_vid.mp4"
        process = subprocess.run(['python3', "run_vd.py", model_str_path, labels, in_vid, output_name, str(cpu_thread), str(probability), "NOGEN"], capture_output=True,text=True)
        result = process.stdout.strip()
        print(f"Average inference time: {result} ms")
        # Store the data in array
        if not "edgetpu" in model_name:
            model_name_arr = model_name.split("_")
            common_model_name = "_".join(model_name_arr[:-2])
            dict["Model"] = common_model_name
            dict["CPU"] = result
        else:
            dict["EdgeTPU"] = result
            inference_results.append(dict.copy())
    # Data is returned in the following format [{Model: value, CPU: value, EdgeTPU: value}, {...}]
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
    print("Draw plots")
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
    plt.savefig("comparison_plot.png", dpi=150)
    print("Script done!")
