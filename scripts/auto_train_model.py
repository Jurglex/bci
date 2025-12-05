import json
from pathlib import Path
import copy
import sys
from neural_decoder.neural_decoder_trainer import trainModel

modelName = 'speechBaseline4'

def load_paths(config_name: str = "paths.json") -> dict:
    config_path = Path(__file__).resolve().parent / config_name
    with config_path.open() as f:
        return json.load(f)

# ----------------------------------------------------------
# Base args
# ----------------------------------------------------------
paths = load_paths()

base_args = {
    'outputDir': str(Path(paths['outputDirBase']) / modelName),
    'datasetPath': paths['datasetPath'],
    'seqLen': 150,
    'maxTimeSeriesLen': 1200,
    'batchSize': 64,
    'lrStart': 0.02,
    'lrEnd': 0.02,
    'nUnits': 1024,
    'nBatch': 10000,
    'nLayers': 5,
    'seed': 0,
    'nClasses': 40,
    'nInputFeatures': 256,
    'dropout': 0.4,
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'gaussianSmoothWidth': 2.0,
    'strideLen': 4,
    'kernelLen': 32,
    'bidirectional': False,
    'l2_decay': 1e-5,
}

# ----------------------------------------------------------
# OVAT hyperparameters
# ----------------------------------------------------------
sweep = {
    "nUnits":       [256, 512],
    "dropout":      [0, 0.05, 0.1, 0.2, 0.3],
    "lrStart":      [0.005, 0.01],
    "nLayers":      [3, 4, 6],
}

# ----------------------------------------------------------
# Run OVAT experiments
# ----------------------------------------------------------
run_id = 0

for param_name, values in sweep.items():
    for val in values:
        run_id += 1

        args = copy.deepcopy(base_args)

        # Set the modified parameter
        args[param_name] = val

        if param_name == "lrStart":
            args["lrEnd"] = val

        # Output directory
        run_name = f"OVAT_{param_name}_{val}_run{run_id:03d}"
        args["outputDir"] = str(Path(base_args["outputDir"]) / run_name)
        Path(args["outputDir"]).mkdir(parents=True, exist_ok=True)

        # Log file path
        log_file = Path(args["outputDir"]) / "log.txt"

        # Print to console and write to log
        print(f"\n=== RUN {run_id}: {param_name} = {val} ===")
        print(f"Saving logs to: {log_file}")

        # Redirect stdout & stderr
        with open(log_file, "w") as f:
            # Duplicate console + file output
            sys.stdout = f
            sys.stderr = f

            print("\n=====================================")
            print(f"RUN {run_id}: {param_name} = {val}")
            print("=====================================")
            print(json.dumps({
                "changed_param": param_name,
                "value": val,
                "outputDir": args["outputDir"]
            }, indent=2))

            # Run training and capture all printed output
            trainModel(args)

        # Restore normal stdout
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        print(f"Finished run {run_id}, logs saved to {log_file}")
