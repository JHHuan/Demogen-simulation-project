from omegaconf import OmegaConf
import hydra
import pathlib


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('demo_generation', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    generator: cls = cls(cfg)
    generator.generate_demo()


if __name__ == "__main__":
    main()
