
modelName = 'speechBaseline4'

import json
from pathlib import Path

def load_paths(config_name: str = "paths.json") -> dict:
    # Resolve paths.json relative to this file
    config_path = Path(__file__).resolve().parent / config_name
    with config_path.open() as f:
        return json.load(f)


args = {}
paths = load_paths()

args['outputDir'] = str(Path(paths['outputDirBase']) / modelName)
args['datasetPath'] = paths['datasetPath']
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel
import torch
trainModel(args)