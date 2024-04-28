import sys, os
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances
from peft import LoraConfig, get_peft_model

from utils import util, parser, commons, test, domain_awareness
from models import vgl_network, dinov2_network
from datasets import gsv_cities, base_dataset


torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, "gsv_cities", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs")

#### Creation of Datasets
logging.debug(f"Loading gsv_cities and {args.dataset_name} from folder {args.datasets_folder}")

if args.domain_awareness:
    train_ds = gsv_cities.GSVCitiesDataset(args)
else:
    if args.use_extra_datasets:
        train_ds = gsv_cities.GSVCitiesDataset(args, cities=(gsv_cities.EXTRA_DATASETS + gsv_cities.TRAIN_CITIES))
    else:
        train_ds = gsv_cities.GSVCitiesDataset(args, cities=gsv_cities.TRAIN_CITIES)

train_dl = DataLoader(train_ds, batch_size= args.train_batch_size, num_workers=args.num_workers, pin_memory= True)

val_ds = base_dataset.BaseDataset(args, "val")
logging.info(f"Val set: {val_ds}")

test_ds = base_dataset.BaseDataset(args, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = vgl_network.VGLNet(args)
model = model.to("cuda")

if args.aggregation == "netvlad":
    train_ds.is_inference = True
    model.aggregation.initialize_netvlad_layer(args, train_ds, model.backbone)
    args.features_dim = args.clusters * dinov2_network.CHANNELS_NUM[args.backbone]
    train_ds.is_inference = False

if args.use_lora:
    lora_modules = ["8.attn.q", "8.atten.k", "8.attn.v", "8.attn.proj", "8.mlp.fc1", "8.mlp.fc2",
                    "9.attn.q", "9.atten.k", "9.attn.v", "9.attn.proj", "9.mlp.fc1", "9.mlp.fc2",
                    "10.attn.q", "10.atten.k", "10.attn.v", "10.attn.proj","10.mlp.fc1", "10.mlp.fc2",
                    "11.attn.q", "11.atten.k", "11.attn.v", "11.attn.proj", "11.mlp.fc1", "11.mlp.fc2",]
    # lora_modules = ["q", "k", "v", "proj", "fc1", "fc2"]
    lora_config = LoraConfig(r=32, lora_alpha=64, use_dora= True, target_modules=lora_modules, lora_dropout=0.01, modules_to_save=["aggregation"])
    model = get_peft_model(model, lora_config)

model = torch.nn.DataParallel(model)
    
util.print_trainable_parameters(model)
util.print_trainable_layers(model)

#### Setup Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
if not args.resume:
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=4000)
criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())
scaler = torch.cuda.amp.GradScaler()

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r1, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1 {best_r1:.1f}")
else:
    best_r1 = start_epoch_num = not_improved_num = 0


if args.domain_awareness:
    domain_awareness.domain_awareness(args, model, train_dl, optimizer, scaler, scheduler, miner, criterion)
    sys.exit()

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses=[]

    model.train()
    for places, labels in tqdm(train_dl, ncols=100):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            features = model(images)
            miner_outputs = miner(features, labels)
            loss = criterion(features, labels, miner_outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not args.resume:
            scheduler.step()

        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, features, miner_outputs, images, labels

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[0] > best_r1
    
    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")
        best_r1 = (recalls[0])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")

    if args.use_lora:
        model.module.save_pretrained(os.path.join(args.save_dir, "lora"))
    
    if not_improved_num == args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break

logging.info(f"Best R@1: {best_r1:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")