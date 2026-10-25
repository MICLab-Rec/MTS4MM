from utils.quick_start import quick_start
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, required=True, help="name of models")
parser.add_argument("--dataset", "-d", type=str, required=True, help="name of datasets")
parser.add_argument("--gpu_id", type=int, default=0, help="id of gpu to use")

args, unknown_args = parser.parse_known_args()
if args.model == 'MTS4MM' and any(arg.split('=', 1)[0] == '--use_kd' for arg in unknown_args):
    parser.error('MTS4MM teachers are fixed on in models/mts4mm.py; --use_kd is no longer supported')

config_dict = {"gpu_id": args.gpu_id}

quick_start(
    model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True
)
