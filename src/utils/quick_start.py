import copy
import json
import os
from itertools import product
from logging import getLogger

import torch
from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import init_seed, get_model, get_trainer, dict2str


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    config = Config(model, dataset, config_dict, mg)
    if config['model'] == 'MTS4MM':
        from models.mts4mm import MTS4MM

        fixed = MTS4MM.FIXED_TEACHER_SETTINGS
        config.final_config_dict.update(fixed)
        config['hyper_parameters'] = [name for name in config['hyper_parameters'] if name not in fixed]
    init_logger(config)
    logger = getLogger()
    dataset_obj = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset_obj.split()
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset,
                                batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset,
                               batch_size=config['eval_batch_size'])
    if 'seed' not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']

    names = list(dict.fromkeys(config['hyper_parameters']))
    choices = []
    for name in names:
        value = config[name]
        choices.append(value if isinstance(value, (list, tuple)) else [value])
    combinations = list(product(*choices))
    if not combinations:
        raise ValueError('The hyperparameter grid has an empty candidate list')
    logger.info('Model: %s; dataset: %s; runs: %d', config['model'], config['dataset'], len(combinations))
    bigger = config['valid_metric_bigger']
    best_score = -float('inf') if bigger else float('inf')
    best_parameters = best_state = best_valid = best_stats = None
    for index, values in enumerate(combinations):
        selected = dict(zip(names, values))
        for name, value in selected.items():
            config[name] = value
        init_seed(config['seed'])
        train_data.pretrain_setup()
        logger.info('Run %d/%d: %s', index + 1, len(combinations), selected)
        instance = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer()(config, instance, mg)
        score, valid_result, _ = trainer.fit(train_data, valid_data=valid_data,
                                             test_data=None, saved=False)
        improved = best_state is None or (score > best_score if bigger else score < best_score)
        if improved:
            best_score, best_parameters = score, copy.deepcopy(selected)
            best_state = trainer.best_model_params
            best_valid = copy.deepcopy(valid_result)
            best_stats = None
            if hasattr(instance, 'stats_history') and hasattr(instance, 'eval_metrics_history'):
                best_stats = (copy.deepcopy(instance.stats_history), copy.deepcopy(instance.eval_metrics_history))
        logger.info('Validation result: %s', dict2str(valid_result))

        del trainer, instance

    for name, value in best_parameters.items():
        config[name] = value
    init_seed(config['seed'])
    instance = get_model(config['model'])(config, train_data).to(config['device'])
    instance.load_state_dict(best_state)
    if best_stats is not None:
        instance.stats_history, instance.eval_metrics_history = best_stats
        if instance.stats_history['epoch']:
            instance.epoch = instance.stats_history['epoch'][-1]
    trainer = get_trainer()(config, instance, mg)
    if save_model:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        prefix = config['model_params_name'] or config['model'].lower()
        if config['model'] == 'MTS4MM' and getattr(instance, 'use_kd', True) is False:
            prefix = prefix.rstrip('_')
            if not prefix.endswith('_student'):
                prefix += '_student'
        checkpoint_path = os.path.join(config['checkpoint_dir'], '{}_{}.pth'.format(prefix, config['dataset']))
        torch.save(best_state, checkpoint_path)

        settings = {key: value for key, value in config.final_config_dict.items()
                    if isinstance(value, (str, int, float, bool, list, tuple)) or value is None}
        with open(checkpoint_path + '.metadata.json', 'w', encoding='utf-8') as stream:
            json.dump({'model': config['model'], 'dataset': config['dataset'],
                       'config': settings, 'best_validation_score': best_score,
                       'best_validation_result': best_valid}, stream, indent=2)

    test_result = trainer.evaluate(test_data, is_test=True)
    logger.info('Selected parameters: %s\nValidation: %s\nTest: %s',
                best_parameters, dict2str(best_valid), dict2str(test_result))
    instance.save_training_stats()
