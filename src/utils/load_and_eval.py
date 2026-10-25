from logging import getLogger, FileHandler, StreamHandler, Formatter
import os
import json
import torch

from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str


def evaluate_from_checkpoint(
    model: str,
    dataset: str,
    config_dict: dict,
    ckpt_path: str,
    mg: bool = False,
    eval_valid: bool = True,
    eval_test: bool = True,
    strict_load: bool = True,
):
    resolved_config = dict(config_dict or {})
    metadata_path = ckpt_path + ".metadata.json"
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["model"] != model or metadata["dataset"] != dataset:
            raise ValueError("Checkpoint metadata does not match requested model/dataset")
        saved_config = metadata["config"]
        saved_config.update(resolved_config)
        resolved_config = saved_config
    config = Config(model, dataset, resolved_config, mg)

    logger = getLogger()
    for h in list(logger.handlers):
        if isinstance(h, FileHandler):
            logger.removeHandler(h)

    if not any(isinstance(handler, StreamHandler) for handler in logger.handlers):
        stream_handler = StreamHandler()
        stream_handler.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)
    logger.setLevel("INFO")
    logger.info("Checkpoint: %s", ckpt_path)

    seed = config["seed"]
    init_seed(seed[0] if isinstance(seed, (list, tuple)) else seed)

    dataset_obj = RecDataset(config)

    train_dataset, valid_dataset, test_dataset = dataset_obj.split()

    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config["train_batch_size"], shuffle=False
    )

    valid_data = None
    test_data = None
    if eval_valid:
        valid_data = EvalDataLoader(
            config,
            valid_dataset,
            additional_dataset=train_dataset,
            batch_size=config["eval_batch_size"],
        )
    if eval_test:
        test_data = EvalDataLoader(
            config,
            test_dataset,
            additional_dataset=train_dataset,
            batch_size=config["eval_batch_size"],
        )

    model_cls = get_model(config["model"])
    if "seed" not in config["hyper_parameters"]:
        config["hyper_parameters"] = ["seed"] + config["hyper_parameters"]
    for i in config["hyper_parameters"]:
        value = config[i]
        if isinstance(value, (list, tuple)):
            config[i] = value[0]
    model_inst = model_cls(config, train_data).to(config["device"])

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=config["device"])

    missing_keys, unexpected_keys = model_inst.load_state_dict(
        state, strict=strict_load
    )
    if (not strict_load) and (missing_keys or unexpected_keys):
        logger.warning(f"[Checkpoint] Missing keys: {missing_keys}")
        logger.warning(f"[Checkpoint] Unexpected keys: {unexpected_keys}")

    trainer = get_trainer()(config, model_inst, mg)

    results = {"valid": None, "test": None}
    with torch.no_grad():
        if valid_data is not None:
            valid_result = trainer.evaluate(valid_data, is_test=False, idx=0)
            results["valid"] = valid_result
            logger.info("VALID result:\n" + dict2str(valid_result))

        if test_data is not None:
            test_result = trainer.evaluate(test_data, is_test=True, idx=0)
            results["test"] = test_result
            logger.info("TEST result:\n" + dict2str(test_result))

    return results, trainer
