import segmentation_models_pytorch as smp
from config import Config
from main import LitUNetPlusPlus
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def main():
    cfg = Config()
    cfg.batch_size = 32
    cfg.epochs = 100
    
    
    model = smp.UnetPlusPlus(**cfg.model_parameters_unetpp)
    unetpp = LitUNetPlusPlus(cfg, model)
    
    wandb_logger = WandbLogger(project="PracticalWorkinAI")
    
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.model_path, monitor='val_loss_total', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss_total', patience=cfg.early_stopping_patience)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(num_nodes=1,
                        devices=4,
                        max_epochs=cfg.epochs, 
                        logger=wandb_logger, 
                        default_root_dir=Path(cfg.default_root_dir),
                        log_every_n_steps=50,
                        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                        deterministic=True)


if __name__ == "__main__":
    main()