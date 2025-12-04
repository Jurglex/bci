"""
Run One-Variable-At-a-Time (OVAT) tests over:
nUnits, dropout, lrStart, nLayers, whiteNoiseSD.

Each run changes exactly ONE parameter, all others remain at base_args values.
"""

import json
from pathlib import Path
import copy
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
# OVAT Hyperparameter Values
# ----------------------------------------------------------

sweep = {
    "nUnits":       [256, 512, 1024],
    "dropout":      [0, 0.05, 0.1, 0.2, 0.3],
    "lrStart":      [0.005, 0.01, 0.02],
    "nLayers":      [3, 4, 5, 6],
    "whiteNoiseSD": [0.0, 0.3, 0.5, 0.8],
}

# ----------------------------------------------------------
# Run OVAT Experiments
# ----------------------------------------------------------
run_id = 0

for param_name, values in sweep.items():
    for val in values:
        run_id += 1

        args = copy.deepcopy(base_args)

        # Change exactly one parameter
        args[param_name] = val

        # Keep lrStart == lrEnd
        if param_name == "lrStart":
            args["lrEnd"] = val

        # Create output folder
        run_name = f"OVAT_{param_name}_{val}_run{run_id:03d}"
        args["outputDir"] = str(Path(base_args["outputDir"]) / run_name)
        Path(args["outputDir"]).mkdir(parents=True, exist_ok=True)

        print("\n=====================================")
        print(f"RUN {run_id}: {param_name} = {val}")
        print("=====================================")
        print(json.dumps({
            "changed_param": param_name,
            "value": val,
            "outputDir": args["outputDir"]
        }, indent=2))

        trainModel(args)
