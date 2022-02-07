import hydra
import omegaconf
import pytorch_lightning as pl
from _solution.common.dispatch import modulename2cls
from _solution.common.get_trainer import get_trainer


@hydra.main(config_path="tasks/flowers/conf", config_name="base")
def main(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))

    # TODO delete this
    config.trainer.wandb = False

    pl.seed_everything(1234)
    module_cls = modulename2cls(name=config.main.module_name)
    model = module_cls(config=config)
    trainer = get_trainer(config=config)
    trainer.fit(model)
    if config.main.is_tested:
        trainer.test(
            model=model,
            ckpt_path="best"
        )

if __name__ == '__main__':
    main()
