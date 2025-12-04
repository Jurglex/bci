import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler_type = args["lr_scheduler"] if ("lr_scheduler" in args) else "linear"
    scheduler = None
    onecycle_scheduler = None
    warmup_scheduler = None
    plateau_scheduler = None
    warmup_steps = 0
    if scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )
    elif scheduler_type == "plateau":
        lr_factor = args["lr_factor"] if ("lr_factor" in args) else 0.5
        lr_patience = args["lr_patience"] if ("lr_patience" in args) else 5
        lr_threshold = args["lr_threshold"] if ("lr_threshold" in args) else 1e-3
        lr_threshold_mode = (
            args["lr_threshold_mode"] if ("lr_threshold_mode" in args) else "rel"
        )
        lr_cooldown = args["lr_cooldown"] if ("lr_cooldown" in args) else 0
        lr_min = args["lr_min"] if ("lr_min" in args) else 1e-6
        lr_eps = args["lr_eps"] if ("lr_eps" in args) else 1e-8
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_threshold,
            threshold_mode=lr_threshold_mode,
            cooldown=lr_cooldown,
            min_lr=lr_min,
            eps=lr_eps,
            verbose=False,
        )
    elif scheduler_type == "two_phase":
        phase1_pct = args["phase1_pct"] if ("phase1_pct" in args) else 0.4
        warmup_steps = int(args["nBatch"] * phase1_pct)
        if warmup_steps < 1:
            warmup_steps = 1
        if warmup_steps >= args["nBatch"]:
            warmup_steps = args["nBatch"] - 1
        phase1_mode = args["phase1_mode"] if ("phase1_mode" in args) else "onecycle"
        if phase1_mode == "onecycle":
            max_lr = args["onecycle_max_lr"] if ("onecycle_max_lr" in args) else args["lrStart"]
            div_factor = args["onecycle_div_factor"] if ("onecycle_div_factor" in args) else 10
            final_div_factor = (
                args["onecycle_final_div_factor"] if ("onecycle_final_div_factor" in args) else 100
            )
            pct_start = args["onecycle_pct_start"] if ("onecycle_pct_start" in args) else 0.1
            onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=warmup_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                cycle_momentum=False,
            )
        else:
            warmup_start_factor = (
                args["warmup_start_factor"] if ("warmup_start_factor" in args) else 0.1
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_steps,
            )
        lr_factor = args["lr_factor"] if ("lr_factor" in args) else 0.5
        lr_patience = args["lr_patience"] if ("lr_patience" in args) else 5
        lr_threshold = args["lr_threshold"] if ("lr_threshold" in args) else 1e-3
        lr_threshold_mode = (
            args["lr_threshold_mode"] if ("lr_threshold_mode" in args) else "rel"
        )
        lr_cooldown = args["lr_cooldown"] if ("lr_cooldown" in args) else 0
        lr_min = args["lr_min"] if ("lr_min" in args) else 1e-6
        lr_eps = args["lr_eps"] if ("lr_eps" in args) else 1e-8
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_threshold,
            threshold_mode=lr_threshold_mode,
            cooldown=lr_cooldown,
            min_lr=lr_min,
            eps=lr_eps,
            verbose=False,
        )

    testLoss = []
    testCER = []
    startTime = time.time()
    train_iter = iter(trainLoader)
    for batch in range(args["nBatch"]):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler_type == "linear" and scheduler is not None:
            scheduler.step()
        elif scheduler_type == "two_phase" and batch < warmup_steps:
            if onecycle_scheduler is not None:
                onecycle_scheduler.step()
            elif warmup_scheduler is not None:
                warmup_scheduler.step()

        # print(endTime - startTime)

        # Eval
        evalInterval = args["evalInterval"] if ("evalInterval" in args) else 100
        if batch % evalInterval == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                if scheduler_type == "plateau" and scheduler is not None:
                    scheduler.step(cer)
                elif scheduler_type == "two_phase" and plateau_scheduler is not None and batch >= warmup_steps:
                    plateau_scheduler.step(cer)

                endTime = time.time()
                eval_interval = evalInterval
                curr_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"batch {batch}, lr: {curr_lr:>8.6f}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/eval_interval:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()