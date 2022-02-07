import numpy as np
import pandas as pd
import hydra
import omegaconf
from _solution.common.dispatch import modulename2cls
from _solution.common.get_trainer import get_trainer

@hydra.main(config_path="../conf", config_name="005_shallow_std")
def main(config: omegaconf.DictConfig) -> None:
    config.trainer.wandb = False
    print(omegaconf.OmegaConf.to_yaml(config))

    list_ckpt = [
        r"C:\c\Users\devoted\Documents\repos\.hydra_outputs\2022-02-02\11-40-16\QedModule\pe8qpcl4\checkpoints\37-0.86.ckpt",
        r"C:\c\Users\devoted\Documents\repos\.hydra_outputs\2022-02-02\12-08-12\QedModule\3fxlqtvz\checkpoints\34-0.88.ckpt",
        r"C:\c\Users\devoted\Documents\repos\.hydra_outputs\2022-02-02\12-37-53\QedModule\1jpxkz2q\checkpoints\37-0.87.ckpt",
        r"C:\c\Users\devoted\Documents\repos\.hydra_outputs\2022-02-02\13-06-39\QedModule\3kem225q\checkpoints\41-0.88.ckpt",
        r"C:\c\Users\devoted\Documents\repos\.hydra_outputs\2022-02-02\13-35-27\QedModule\3r8cv8qp\checkpoints\39-0.88.ckpt"

    ]
    list_outputs = []

    module_cls = modulename2cls(name=config.main.module_name)
    model = module_cls(config=config, handle_output=list_outputs)
    trainer = get_trainer(config=config)


    for ckpt_path in list_ckpt:
        trainer.test(
            model=model,
            ckpt_path=ckpt_path
        )

    df = pd.DataFrame(np.stack(list_outputs).T)
    df.to_csv(r"C:\temp\qed\results\cv_net.csv", index=False)


if __name__ == '__main__':
    main()
