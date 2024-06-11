
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from lightning_module import LitUNetPlusPlus
from config import Config
from dataset import *
import wandb
import argparse
from utils import str2bool
from monai.transforms import *
import glob

from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False


def run(source_domains, experiment_name, 
        batch_size, epochs, use_official_testset, 
        loss_smoothing, train_split=0.8, val_split=0.2, 
        test_split=0.1, exp_with_svdna=True, 
        with_histogram=True, histogram_matching_only=False):
    
    seed_everything(99, workers=True)

    # DEFINE EXPERIMENT PARAMETERS
    ####################################################################################
    cfg = Config(
        source_domains=source_domains,
        experiment_name=experiment_name,
        batch_size=batch_size,
        epochs=epochs,
        use_official_testset=use_official_testset,
        loss_smoothing=loss_smoothing,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        exp_with_svdna=exp_with_svdna,
        with_histogram=with_histogram,
        histogram_matching_only=histogram_matching_only
    )
    ####################################################################################


    # OCTDatasetPrep is a class that performs all necessary sorting and filtering of the dataset
    ds = OCTDatasetPrep(cfg.train_dir, source_domains = cfg.source_domains)

    # .get_datasets method handles splitting and which test set to use (a custom one that is split from
    # the training set, or the official challenge testset)
    train_data, val_data, test_data = ds.get_datasets(dataset_split=[cfg.train_split, cfg.val_split, cfg.test_split], use_official_testset=cfg.use_official_testset)

    # MakeDataset is a class that handles the actual dataset creation, including transforms
    train_dataset = MakeDataset(train_data, cfg.train_transforms)
    val_dataset = MakeDataset(val_data, cfg.val_transforms)
    test_dataset = MakeDataset(test_data, cfg.test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=7, persistent_workers=True)
    
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=7, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, 
                             num_workers=7, persistent_workers=True)


    wandb_logger = WandbLogger(project="PracticalWorkinAI", 
                               name=cfg.experiment_name)


    experiment_folder = Path("/home/optima/mhaderer/OPTIMA_Masterarbeit/practical/models/") / cfg.experiment_name
    checkpoint = list(experiment_folder.glob("*.ckpt"))[0]

    model = LitUNetPlusPlus.load_from_checkpoint(checkpoint_path = checkpoint, experiment_name=cfg.experiment_name)
    
    # model is the actual model from segmentation_models_pytorch
    #model = smp.UnetPlusPlus(**cfg.model_parameters_unetpp)

    # LitUnetPlusPlus is the lightning module defined in lightning_module.py
    unetpp = LitUNetPlusPlus(cfg, model, experiment_name=cfg.experiment_name)

    trainer = L.Trainer(max_epochs=cfg.epochs,
                        logger=wandb_logger,
                        default_root_dir=Path(cfg.default_root_dir),
                        log_every_n_steps=10,
                        deterministic=True,
                        callbacks=[cfg.checkpoint_callback, # saves best model based on validation loss, every 5 epochs
                                    cfg.lr_monitor, 
                                    #cfg.save_initial_model,
                                    #cfg.early_stopping, # stops training if validation loss does not improve for 20 epochs
                                    cfg.aggregate_testing_results])


    trainer.fit(unetpp, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(unetpp, dataloaders=test_loader)
    wandb.finish


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script runs a specified experiment using SVDNA on domain shifted data.')
    # required args
    parser.add_argument('--source_domains', nargs='+', required=True, help='List of source domains')
    parser.add_argument('--experiment_name', required=True, help='Name of the experiment')
    parser.add_argument('--exp_with_svdna', type=str, required=True, help='Use SVDNA (True/False)')

    # args for ablation
    parser.add_argument('--with_histogram', type=str, default="True", help='Can turn on/turn off histogram matching (True/False)')
    parser.add_argument('--histogram_matching_only', type=str, default="True", help='ONLY perform histogram matching, no noise adaptation. (True/False)')

    # args with defaults
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--use_official_testset', type=str, default="True", help='Use official testset')
    parser.add_argument('--loss_smoothing', type=float, default=1e-5, help='Loss smoothing')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')

    args = parser.parse_args()

    # for easier handling of bools, convert any of "y, Yes, True, 1" to bool: True and viceversa False
    use_official_testset = str2bool(args.use_official_testset)
    exp_with_svdna = str2bool(args.exp_with_svdna)
    with_histogram = str2bool(args.with_histogram)
    histogram_matching_only = str2bool(args.histogram_matching_only)

    run(args.source_domains, 
        args.experiment_name, 
        args.batch_size, 
        args.epochs, 
        use_official_testset, 
        args.loss_smoothing,
        args.train_split,
        args.val_split,
        args.test_split,
        exp_with_svdna,
        with_histogram,
        histogram_matching_only)