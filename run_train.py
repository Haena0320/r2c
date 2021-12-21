import argparse
import os, sys
sys.path.append('/mnt/data/user8/vision_data/r2c/r2c_pytorch')
import shutil

import multiprocessing
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from src.model import *
from src.dataloader import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, restore_checkpoint, restore_best_checkpoint
from utils.utils import *

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--config',default="/mnt/data/user8/vision_data/r2c/r2c_pytorch/src/config.json",help='Params location',type=str)
parser.add_argument('--rationale',action="store_true",help='use rationale')
parser.add_argument('--folder',dest='folder',help='folder location',type=str)
parser.add_argument('--weight_decay',default=0.0001,type=float)
parser.add_argument('--learning_rate',default=0.0002,type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--warmup_proportion',default=0.1, type=float)
parser.add_argument('--adam_betas',default=None)
parser.add_argument('-no_tqdm',dest='no_tqdm',action='store_true')

args = parser.parse_args()
config = load_config(args.config)
train, val, test = VCR.splits()

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
NUM_GPUS = 1
if NUM_GPUS==0:
    raise ValueError("you need gpus")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td

# num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
# print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
num_workers = 1
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100

model = AttentionQA(config.model)
for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False


model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
args.t_total = len(train_loader) * config.trainer.num_epochs
args.warmup_steps = args.t_total * args.warmup_proportion
model, optimizer, scheduler = setup_opt(args, model)

log_folder = os.path.join(args.folder, "log")
if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, save_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)

else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    os.makedirs(log_folder)
    start_epoch, val_metric_per_epoch = 0, []

log_writer = SummaryWriter(log_folder)


num_batches = 0
for epoch_num in range(start_epoch, config['trainer']['num_epochs'] + start_epoch):
    train_results = []
    norms = []
    model.train()
    import time
    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch)
        optimizer.zero_grad()
        output_dict = model(**batch)
        loss = output_dict['loss'].mean() + output_dict['cnn_regularization_loss'].mean()
        loss.backward()

        num_batches += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        scheduler.step()

        train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                        'crl': output_dict['cnn_regularization_loss'].mean().item(),
                                        'accuracy': output_dict["accuracy"],
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))
        log_writer.add_scalar("train/loss", output_dict['loss'].mean().item(), num_batches)
        log_writer.add_scalar("train/crl_loss", output_dict['cnn_regularization_loss'].mean().item(), num_batches)
        log_writer.add_scalar("train/accuracy", output_dict["accuracy"], num_batches)
        log_writer.add_scalar("train/sec_per_batch", time_per_batch, num_batches)
        log_writer.add_scalar("train/hr_per_epoch", len(train_loader) * time_per_batch / 3600, num_batches)

        # if b % ARGS_RESET_EVERY == 0 and b > 0:
        #     norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
        #         param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)
        #
        #     print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
        #         epoch_num, b, len(train_loader),
        #         norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
        #         pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
        #     ), flush=True)

    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
    val_probs = []
    val_labels = []
    val_loss_sum = 0.0
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
    print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
          flush=True)
    log_writer.add_scalar("val/acc",val_metric_per_epoch[-1], epoch_num)
    log_writer.add_scalar("val/loss", val_loss_avg, epoch_num)


    save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        output_dict = model(**batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.3f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)

