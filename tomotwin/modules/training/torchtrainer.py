"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

import copy
import os
from typing import Any, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import tomotwin
from tomotwin.modules.common import preprocess
from tomotwin.modules.networks.torchmodel import TorchModel
from tomotwin.modules.networks.networkmanager import NetworkManager
from tomotwin.modules.training.trainer import Trainer
from tomotwin.modules.training.tripletdataset import TripletDataset


class TorchTrainer(Trainer):
    """
    Trainer for pytorch.
    """

    def __init__(
        self,
        epochs: int,
        batchsize: int,
        learning_rate: float,
        network: TorchModel,
        criterion: nn.Module,
        decoder_criterion: callable = None,
        training_data: TripletDataset = None,
        test_data: TripletDataset = None,
        workers: int = 0,
        output_path: str = None,
        log_dir: str = None,
        checkpoint: str = None,
        optimizer: str = "Adam",
        amsgrad: bool = False,
        weight_decay: float = 0,
        patience: int = None,
        save_epoch_seperately: bool = False,
        train_with_triplet_loss: bool = True,
        train_with_reconstruction_loss: bool = False,
    ):
        """
        :param epochs: Number of epochs
        :param batchsize: Training batch size
        :param learning_rate: The learning rate
        """

        super().__init__()

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.local_rank = 0
        
        cudnn.benchmark = True
        self.epochs = epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.test_data = test_data
        self.patience = patience
        self.train_with_triplet_loss = train_with_triplet_loss
        self.train_with_reconstruction_loss = train_with_reconstruction_loss
        if self.patience is None:
            self.patience = self.epochs
        self.workers = workers
        self.best_model_loss = None
        self.best_model_f1 = None
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.criterion = criterion
        self.decoder_criterion = decoder_criterion
        self.network = network
        self.network_config = None
        self.output_path = output_path
        self.last_loss = None
        self.best_val_loss = np.Infinity

        self.best_val_f1 = 0
        self.current_epoch = None
        self.best_epoch_loss = None
        self.best_epoch_f1 = None
        self.checkpoint = None
        self.start_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.network.init_weights()
        self.model = self.network.get_model()
        self.checkpoint = checkpoint
        self.save_epoch_seperately = save_epoch_seperately
        self.f1_improved = False
        self.loss_improved = False

        # Write graph to tensorboard
        if not dist.is_initialized() or dist.get_rank() == 0:
            dummy_input = torch.zeros([12, 1, 37, 37, 37], device=self.device)
            try:
                self.writer.add_graph(self.model, dummy_input)
            except Exception as e:
                print(f"Could not add graph to TensorBoard: {e}")

        self.model = self.model.to(self.device)
        
        optimizer_class = getattr(optim, optimizer)
    
        additional_optimizer_params = {"weight_decay": weight_decay}
        if "Adam" in optimizer_class.__name__:
            additional_optimizer_params["amsgrad"] = amsgrad
        
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            **additional_optimizer_params
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience, verbose=True
        )
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        self.run_if_rank_0(
            print, 
            "Number of parameters:", params
            )

        self.run_if_rank_0(
        self.writer.add_text,
            "Optimizer", 
            type(self.optimizer).__name__
        )
        self.run_if_rank_0(
            self.writer.add_text,
            "Initial learning rate",
            str(self.learning_rate)
        )

        if self.checkpoint is not None:
            self.load_checkpoint(checkpoint=self.checkpoint)



        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model.to(self.local_rank),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = self.model.to(self.device)
    
    def run_if_rank_0(self, func, *args, **kwargs):
        """
        Run a function only if not in DDP mode.
        This is useful for functions that should not be run in DDP mode, e.g. self.run_if_not_ddping.
        print,
        """
        if not dist.is_initialized() or dist.get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
            
    def set_seed(self, seed: int):
        """
        Set the seed for random number generators
        """
        torch.manual_seed(seed)
        torch.seed()

    def get_train_test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create a dataloaders for the train and validation data
        """
        train_sampler = None
        test_sampler = None

        if torch.distributed.is_initialized():
            train_sampler = DistributedSampler(self.training_data, shuffle=True)
            if self.test_data is not None:
                test_sampler = DistributedSampler(self.test_data, shuffle=False)
            
        shuffle_kwarg = {"shuffle": True} if train_sampler is None else {}
        train_loader = DataLoader(
            self.training_data,
            batch_size=self.batchsize,
            sampler=train_sampler,
            num_workers=self.workers,
            pin_memory=False,
            # prefetch_factor=5,
            timeout=180,
            **shuffle_kwarg,
        )

        test_loader = None
        shuffle_kwarg = {"shuffle": False} if test_sampler is None else {}
        if self.test_data is not None:
            test_loader = DataLoader(
                self.test_data,
                batch_size=self.batchsize,
                sampler=test_sampler,
                num_workers=self.workers,
                pin_memory=False,
                # prefetch_factor=5,
                timeout=60,
                **shuffle_kwarg,
            )
        return train_loader, test_loader

    @staticmethod
    def get_best_f1(
        anchor_label: str, similarities: np.array, sim_labels: Iterable
    ) -> Tuple[float, float]:
        """
        Caluclate the classification F1 score for a given anchor
        """
        PDB = os.path.splitext(anchor_label)[0].upper()
        gt_mask = np.array([PDB in p.upper() for p in sim_labels])
        best_f1 = 0
        best_t = None
        for t in np.arange(0, 1, 0.025):
            picked = similarities > t

            true_positive = np.logical_and(gt_mask, picked)
            TP = np.sum(true_positive)
            false_positive = np.logical_and(gt_mask == False, picked)
            FP = np.sum(false_positive)
            false_negative = np.logical_and(gt_mask, picked == False)
            FN = np.sum(false_negative)
            f1 = 2 * TP / (2 * TP + FP + FN)
            if f1 >= best_f1:
                best_t = t
                best_f1 = f1

        return best_f1, best_t

    @staticmethod
    def calc_avg_f1(anchors: pd.DataFrame, volumes: pd.DataFrame) -> float:
        """
        Calculates average f1 score
        Each column in 'anchors' represents an anchor volume.
        Each column in 'volumes' represents an tomogram subvolume
        :return: Classification accuracy
        """
        scores = []
        for col in anchors:
            sim = np.matmul(volumes.T, anchors[col])
            best_f1, _ = TorchTrainer.get_best_f1(
                anchor_label=col, similarities=sim, sim_labels=sim.index.values
            )
            scores.append(best_f1)
        avg_f1 = np.mean(scores)
        return avg_f1

    def classification_f1_score(self, test_loader: DataLoader) -> float:
        """
        Calculates classification f1 score
        :return: F1 score
        """
        self.model.eval()
        disable_bar = dist.is_initialized() and dist.get_rank() != 0
        t = tqdm(test_loader, desc="Classification accuracy", leave=False, disable=disable_bar)
        anchor_emb = {}  # pd.DataFrame()
        vol_emb = {}  # pd.DataFrame()

        with torch.no_grad():
            for _, batch in enumerate(t):
                anchor_vol = batch["anchor"]
                positive_vol = batch["positive"]
                negative_vol = batch["negative"]
                full_input = torch.cat((anchor_vol,positive_vol,negative_vol), dim=0).to(self.device, non_blocking=True)
                filenames = batch["filenames"]
                with autocast():
                    out = self.model.forward(full_input)
                    out = torch.split(out, anchor_vol.shape[0], dim=0)
                    anchor_out = out[0]
                    positive_out = out[1]
                    negative_out = out[2]                    

                    anchor_out_np = anchor_out.cpu().detach().numpy()
                    for i, anchor_filename in enumerate(filenames[0]):
                        if preprocess.label_filename(anchor_filename) not in anchor_emb:
                            anchor_emb[
                                preprocess.label_filename(anchor_filename)
                            ] = anchor_out_np[i, :]
                    positive_out_np = positive_out.cpu().detach().numpy()
                    for i, pos_filename in enumerate(filenames[1]):
                        if os.path.basename(pos_filename) not in vol_emb:
                            vol_emb[os.path.basename(pos_filename)] = positive_out_np[
                                i, :
                            ]

                    negative_out_np = negative_out.cpu().detach().numpy()
                    for i, neg_filename in enumerate(filenames[2]):
                        if os.path.basename(neg_filename) not in vol_emb:
                            vol_emb[os.path.basename(neg_filename)] = negative_out_np[
                                i, :
                            ]

        return TorchTrainer.calc_avg_f1(pd.DataFrame(anchor_emb), pd.DataFrame(vol_emb))

    def run_batch(self, batch: Dict, mode: str):
        """
        Run inference on one batch.
        :param batch: Dictionary with batch data
        :return: Loss of the batch
        """
        losses = {}
        
        anchor_vol = batch["anchor"]
        positive_vol = batch["positive"]
        negative_vol = batch["negative"]
        full_input = torch.cat((anchor_vol,positive_vol,negative_vol), dim=0).to(self.device, non_blocking=True)
        with autocast():
            # get embeddings and ignore decoder output which may or may not be present
            out = self.model.forward(full_input)
            if (isinstance(out, tuple) or isinstance(out, list)) and len(out) == 2:
                out = out[0]
                
        out = torch.split(out, anchor_vol.shape[0], dim=0)
        
        if mode == "val" or (mode == "train" and self.train_with_triplet_loss):
            triplet_loss = self.criterion(
                out[0],
                out[1],
                out[2],
                label_anchor=batch["label_anchor"],
                label_positive=batch["label_positive"],
                label_negative=batch["label_negative"],
            )
        elif mode == "train" and not self.train_with_triplet_loss:
            triplet_loss = torch.tensor(0.0, device=self.device)
        loss = triplet_loss        

        if mode == "train" and self.train_with_reconstruction_loss:
            anchor_vol_even = batch["anchor_even"]
            anchor_vol_odd = batch["anchor_odd"]
            full_input = torch.cat([anchor_vol_even], dim=0).to(self.device, non_blocking=True)
            with autocast():
                anchor_vol_even_enc, anchor_vol_even_dec = self.model.forward(full_input)
            decoder_loss = self.decoder_criterion(
                anchor_vol_even_dec,
                anchor_vol_odd.to(anchor_vol_even_dec.device, non_blocking=True),
            )
            loss = (loss + decoder_loss) / 2.0
            losses["decoder_out"] = decoder_loss
            losses["triplet_loss"] = triplet_loss
            
        # if mode == "train" and self.train_with_reconstruction_loss and not self.train_with_triplet_loss:
        #     consistency_target = out[0].clone().detach() 
        #     consistency_loss = (1-(consistency_target@anchor_vol_even_enc.T).diag()).mean()
        #     losses["consistency_loss"] = consistency_loss
        #     loss = ((loss * 2.0) + consistency_loss) / 3.0
        
        losses["loss"] = loss

        
        return losses

    def save_best_loss(self, current_val_loss: float, epoch: int) -> None:
        """
        Update best model according loss
        :param current_val_loss: Current validation loss
        :param epoch: Current epoch
        :return:  None
        """
        if current_val_loss < self.best_val_loss:
            self.loss_improved = True
            self.run_if_rank_0(
                print,
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}"
            )
            self.best_epoch_loss = epoch
            self.best_val_loss = current_val_loss
            self.best_model_loss = copy.deepcopy(self.model)

    def save_best_f1(self, current_val_f1: float, epoch: int) -> None:
        """
        Update best model according f1 one score
        :param current_val_f1: Current f1 score
        :param epoch: Current epoch
        :return: None
        """
        if current_val_f1 > self.best_val_f1:
            self.f1_improved = True
            self.run_if_rank_0(
                print,
                f"Validation F1 score improved from {self.best_val_f1} to {current_val_f1}"
            )
            self.best_epoch_f1 = epoch
            self.best_val_f1 = current_val_f1
            self.best_model_f1 = copy.deepcopy(self.model)

    def validation_loss(self, test_loader: DataLoader) -> float:
        """
        Runs the current model on the validation data
        :return: Validation loss
        """
        val_loss = []
        self.model.eval()
        disable_bar = dist.is_initialized() and dist.get_rank() != 0
        t = tqdm(test_loader, desc="Validation", leave=False, disable=disable_bar)

        try:
            self.criterion.set_validation(True)
        except:
            self.run_if_rank_0(
                print, 
                "Cant activate validation mode for loss"
            )
        with torch.no_grad():
            for _, batch in enumerate(t):
                valloss = self.run_batch(batch, mode="val")["loss"]
                val_loss.append(valloss.cpu().detach().numpy())
                desc_t = f"Validation (running loss: {np.mean(val_loss[-20:]):.4f} "
                t.set_description(desc=desc_t)
        try:
            self.criterion.set_validation(False)
        except:
            self.run_if_rank_0(
                print, 
                "Cant deactivate validation mode for loss"
            )

        current_val_loss = np.mean(val_loss)
        return current_val_loss

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        Load model checkpoint
        :param checkpoint: Path to checkpoint
        :return: None
        """

        try:
            self.checkpoint = torch.load(checkpoint)
        except FileNotFoundError:
            self.run_if_rank_0(
                print, 
                f"Checkpoint {checkpoint} can't be found. Ignore it."
            )
            self.checkpoint = None
            return

        state_dict = self.checkpoint["model_state_dict"]
        state_dict = NetworkManager.remove_module_prefix(state_dict)
        self.model.load_state_dict(state_dict)
        
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        self.start_epoch = self.checkpoint["epoch"] + 1
        self.last_loss = self.checkpoint["loss"]
        self.best_val_loss = self.checkpoint["best_loss"]
        self.best_val_f1 = self.checkpoint["best_f1"]
        self.run_if_rank_0(
            print,
            f"Restart from checkpoint. Epoch: {self.start_epoch}, Training loss: {self.last_loss}, Validation loss: {self.best_val_loss}"
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Runs a single epoch
        :param train_loader: Data loader for training data
        :return: Training loss after the epoch
        """

        scaler = GradScaler()
        running_loss = []
        running_triplet_loss = []
        running_decoder_loss = []
        running_consistency_loss = []
        
        self.model.train()
        disable_bar = dist.is_initialized() and dist.get_rank() != 0
        t = tqdm(train_loader, desc="Training", leave=False, disable=disable_bar)
        for _, batch in enumerate(t):
                        
            self.optimizer.zero_grad()

            losses = self.run_batch(batch, mode="train")
            
            loss = losses["loss"]
            
            loss_np = loss.cpu().detach().numpy()
            running_loss.append(loss_np)
            
            if "triplet_loss" in losses:
                triplet_loss_np = losses["triplet_loss"].cpu().detach().numpy()
                running_triplet_loss.append(triplet_loss_np)
            
            if "decoder_out" in losses:
                decoder_loss_np = losses["decoder_out"].cpu().detach().numpy()
                running_decoder_loss.append(decoder_loss_np)
                
            if "consistency_loss" in losses:
                consistency_loss_np = losses["consistency_loss"].cpu().detach().numpy()
                running_consistency_loss.append(consistency_loss_np)
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            desc_t = f"Training (loss: {np.mean(running_loss[-20:]):.4f}) "

            t.set_description(desc=desc_t)

        training_loss = np.mean(running_loss)
        out = {"train_loss": training_loss}
        self.last_loss = training_loss

        if "triplet_loss" in losses:
            out["train_triplet_loss"] = np.mean(running_triplet_loss)
        if "decoder_out" in losses:
            out["train_decoder_loss"] = np.mean(running_decoder_loss)
        if "consistency_loss" in losses:
            out["train_consistency_loss"] = np.mean(running_consistency_loss)
        
        return out

    def train(self) -> nn.Module:
        """
        Trains the model and returns it.
        :return: Trained model
        """
        if self.training_data is None:
            raise RuntimeError("Training data is not set")

        train_loader, test_loader = self.get_train_test_dataloader()

        self.current_epoch = -1
        if self.train_with_reconstruction_loss and not self.train_with_triplet_loss:
            self.validate(test_loader=test_loader)
        
        # Training Loop
        disable_bar = dist.is_initialized() and dist.get_rank() != 0
        for epoch in tqdm(
            range(self.start_epoch, self.epochs),
            initial=self.start_epoch,
            total=self.epochs,
            desc="Epochs",
            disable=disable_bar,
        ):
            if dist.is_initialized() and hasattr(train_loader, "sampler"):
                train_loader.sampler.set_epoch(epoch)
            self.f1_improved = False
            self.loss_improved = False
            self.current_epoch = epoch
            
            if self.train_with_reconstruction_loss:
                model_ref = self.model.module if hasattr(self.model, "module") else self.model
                model_ref.decode = True

            train_losses = self.train_epoch(train_loader=train_loader)
            train_loss = train_losses["train_loss"]

            self.run_if_rank_0(
                print, 
                f"Epoch: {epoch + 1}/{self.epochs} - Training Loss: {train_loss:.4f}"
                )
            self.run_if_rank_0(
                self.writer.add_scalar,
                "Loss/train",
                train_loss,
                epoch,
            )
            
            if "train_triplet_loss" in train_losses:
                self.run_if_rank_0(
                    self.writer.add_scalar,
                    "Loss/train_triplet", train_losses["train_triplet_loss"], epoch
                )
            if "train_decoder_loss" in train_losses:
                self.run_if_rank_0(
                    self.writer.add_scalar,
                    "Loss/train_decoder", train_losses["train_decoder_loss"], epoch
                )
            if "train_consistency_loss" in train_losses:
                self.run_if_rank_0(
                    self.writer.add_scalar,
                    "Loss/train_consistency", train_losses["train_consistency_loss"], epoch
                )

            # Validation
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            model_ref.decode = False
            
            if test_loader is not None:
                self.validate(test_loader=test_loader)

            self.run_if_rank_0(self.writer.flush)

            if self.output_path is not None:
                self.run_if_rank_0(
                    self.write_results_to_disk,
                    self.output_path,
                    save_each_improvement=self.save_epoch_seperately,
                )

        return self.model
    
    def validate(self, test_loader: DataLoader) -> Tuple[float, float]:
        # Validation
        if test_loader is not None:
            current_val_loss = self.validation_loss(test_loader)
            current_val_f1 = self.classification_f1_score(test_loader=test_loader)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.scheduler.step(current_val_loss)
            self.run_if_rank_0(
                print, 
                f"Validation Loss: {current_val_loss:.4f}."
            )
            self.run_if_rank_0(
                print, 
                f"Validation F1 Score: {current_val_f1:.4f}."
            )
            self.run_if_rank_0(
                self.writer.add_scalar,
                "Loss/validation", 
                current_val_loss, 
                self.current_epoch
            )
            self.run_if_rank_0(
                self.writer.add_scalar,
                "F1/validation", 
                current_val_f1, 
                self.current_epoch
            )
            self.save_best_loss(current_val_loss, self.current_epoch)
            self.save_best_f1(current_val_f1, self.current_epoch)

    def set_training_data(self, training_data: TripletDataset) -> None:
        """
        Set the training data
        """
        self.training_data = training_data

    def set_test_data(self, test_data: TripletDataset) -> None:
        """
        Set test (validation) data.
        """
        self.test_data = test_data

    def set_network_config(self, config) -> None:
        """
        Set the network config
        """
        self.network_config = config

    @staticmethod
    def _write_model(
        path: str,
        model: TorchModel,
        config: Dict,
        optimizer=None,
        loss: float = None,
        epoch: int = None,
        best_loss: float = None,
        best_f1: float = None,
        **kwargs,
    ):
        """
        Adds some metadata to the model and write the model  to disk

        :param path: Path where the model should be written
        :param model: The model that is saved to disk
        :param config: Configuration of tomotwin
        :param optimizer: Optimizer
        :param loss: Loss
        :param epoch: Current epoch
        :param best_loss: Current best validation loss
        :param best_f1:  Current best validtion f1 score
        :return:
        """
        for key, value in kwargs.items():
            config[key] = value
        results_dict = {
            "model_state_dict": model.state_dict(),
            "tomotwin_config": config,
            "tt_version_train": tomotwin.__version__,
        }
        if optimizer is not None:
            results_dict["optimizer_state_dict"] = optimizer.state_dict()

        if loss is not None:
            results_dict["loss"] = loss

        if best_loss is not None:
            results_dict["best_loss"] = best_loss

        if best_f1 is not None:
            results_dict["best_f1"] = best_f1

        if epoch is not None:
            results_dict["epoch"] = epoch

        torch.save(
            results_dict,
            path,
        )

    def write_model_to_disk(
        self, path: str, model_to_save, model_name: str, epoch: int, **kwargs
    ):
        """
        :param path: Path for folder where the model is saved to.
        :param model_to_save:  Model to save
        :param model_name: model filename
        :param epoch: Epoch of the model
        :return: None
        """
        if isinstance(model_to_save, nn.DataParallel):
            model_to_save = model_to_save.module

        self._write_model(
            path=os.path.join(path, model_name),
            model=model_to_save,
            config=self.network_config,
            optimizer=self.optimizer,
            loss=self.last_loss,
            best_loss=self.best_val_loss,
            best_f1=self.best_val_f1,
            epoch=epoch,
            **kwargs,
        )

    def write_results_to_disk(
        self, path: str, save_each_improvement: bool = False, **kwargs
    ) -> None:
        """
        Write the training results to specified folder
        :param path: Path to folder to write the data
        :param save_each_improvement: If true, model for each epoch is saved.
        :param kwargs:
        :return: None
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self.write_model_to_disk(path, self.model, "latest.pth", self.current_epoch)

        if self.current_epoch == self.epochs - 1:
            if os.path.exists(os.path.join(path, "final.pth")):
                os.remove(os.path.join(path, "final.pth"))
            os.rename(os.path.join(path, "latest.pth"), os.path.join(path, "final.pth"))

        if self.best_model_loss is not None and self.loss_improved:
            # The best_model can be None, after a training restart.
            print(f"Saving best model according to loss")
            self.write_model_to_disk(
                path,
                self.best_model_loss,
                "best_loss.pth",
                self.best_epoch_loss,
                **kwargs,
            )

        if self.best_model_f1 is not None and self.f1_improved:
            print(f"Saving best model according to f1")
            # The best_model can be None, after a training restart.
            self.write_model_to_disk(
                path, self.best_model_f1, "best_f1.pth", self.best_epoch_f1, **kwargs
            )

        if save_each_improvement:
            mod = self.model
            ep = self.current_epoch
            add = "" + "_f1" if self.f1_improved else ""
            add = add + "_loss" if self.loss_improved else ""
            if self.f1_improved or self.loss_improved:
                self.write_model_to_disk(
                    path,
                    mod,
                    "best_model_" + f"{ep + 1}".zfill(3) + f"{add}.pth",
                    ep,
                    **kwargs,
                )

    def get_model(self) -> Any:
        """
        :return: Trained model
        """
        return self.model
