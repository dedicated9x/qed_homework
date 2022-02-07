import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_trainer(config):
    # Checkpoints
    if config.trainer.monitored_metric is not None:
        metric_name = config.trainer.monitored_metric.name
        callbacks = [ModelCheckpoint(
            monitor=metric_name,
            filename='{epoch}-' + f'{{{metric_name}:.2f}}',
            mode=config.trainer.monitored_metric.mode,
            auto_insert_metric_name=("/" not in metric_name)
        )]
    else:
        callbacks = None

    # Wandb
    if config.trainer.wandb:
        logger = WandbLogger(
            project=config.main.module_name,
            name=config.main.module_name.lower(),
        )
    else:
        logger = True

    trainer = pl.Trainer(
        gpus=config.trainer.device,
        max_epochs=config.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=config.trainer.limit_train_batches
    )

    return trainer