from tqdm import tqdm
import math
import argparse
import torch
from timm.utils import ModelEmaV2

import cloc
from cloc.training import set_logging, set_device, set_model, adjust_lr, vr_evaluate, save_checkpoints


from tqdm import tqdm
import torch
import cloc

def compute_loss_per_image(model, dataset, device):
    loader = cloc.datasets.make_val_loader(dataset, batch_size=1, workers=1)
    model.eval()
    losses = []
    indices = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc='Computing per-image losses'), start=1):
            # data[0] là tensor ảnh
            image = data[0].to(device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            loss = model(image)['loss'].item()

            losses.append(loss)
            # chỉ thêm idx nếu chưa có (đề phòng duplicate)
            if idx not in indices:
                indices.append(idx)

    return indices, losses


def select_top_k_percent(indices, losses, percent=0.7):
    combined = list(zip(indices, losses))
    combined.sort(key=lambda x: x[1], reverse=True)
    k = int(len(combined) * percent)
    top_indices = [idx for idx, _ in combined[:k]]
    return top_indices


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    boa = argparse.BooleanOptionalAction
    # wandb setting
    parser.add_argument('--wbproject',  type=str,   default='cloc')
    parser.add_argument('--wbgroup',    type=str,   default='pre-train-vr')
    parser.add_argument('--wbmode',     type=str,   default='disabled')
    parser.add_argument('--name',       type=str,   default=None)
    # model setting
    parser.add_argument('--model',      type=str,   default='our_model')
    parser.add_argument('--model_args', type=str,   default='')
    # data setting
    parser.add_argument('--trainset',   type=str,   default='coco_train2017')
    parser.add_argument('--valset',     type=str,   default='kodak')
    # optimization setting
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--iterations', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--lr_sched',   type=str,   default='const-0.5-cos')
    parser.add_argument('--grad_clip',  type=float, default=2.0)
    parser.add_argument('--ema_decay',  type=float,   default=0.9999)
    # training acceleration and device setting
    parser.add_argument('--compile',    action=boa, default=False)
    parser.add_argument('--workers',    type=int,   default=1)
    cfg = parser.parse_args()

    # default settings
    cfg.wandb_log_interval = 20
    cfg.model_val_interval = 2_000
    return cfg


def main():
    cfg = parse_args()

    log_dir, wbrun = set_logging(cfg)
    device = set_device()

    model = set_model(cfg)
    model = model.to(device)

    model_ema = ModelEmaV2(model, decay=cfg.ema_decay)
    model_cmp = torch.compile(model) if cfg.compile else model

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    trainset = cloc.get_dataset(cfg.trainset)
    trainloader = cloc.datasets.make_train_loader(trainset, cfg.batch_size, cfg.workers)
    valset = cloc.get_dataset(cfg.valset)
    valloader = cloc.datasets.make_val_loader(valset, batch_size=1, workers=0)

    # ======================== training loops ========================
    pbar = tqdm(range(0, cfg.iterations), ascii=True)
    for step in pbar:
        if (step > 0) and (step % cfg.model_val_interval == 0): # evaluation
            vr_evaluate(model_ema, valloader, wbrun, step)
            save_checkpoints(log_dir, step, model, model_ema, optimizer)
            model.train()

        if step % 10 == 0: # learning rate schedule
            adjust_lr(cfg, optimizer, step)

        # training step
        assert model_cmp.training
        batch = next(trainloader).to(device=device)
        metrics = model_cmp(batch)

        metrics['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_cmp.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        model_ema.decay = cfg.ema_decay * (1 - math.exp(-step / 10_000)) # warmup from 0 to cfg.ema_decay
        model_ema.update(model)

        # logging
        msg = ', '.join([f'{k}={float(v):.4f}' for k, v in metrics.items()])
        pbar.set_description(msg)
        if step % cfg.wandb_log_interval == 0: # log to wandb
            log_dict = {f'train/{k}': float(v) for k, v in metrics.items()}
            log_dict['general/grad_norm'] = grad_norm.item()
            for i, pg in enumerate(optimizer.param_groups):
                log_dict[f'general/lr-pg{i}'] = pg['lr']
            wbrun.log(log_dict, step=step)

    # final evaluation
    #vr_evaluate(model_ema, valloader, wbrun, step)
    save_checkpoints(log_dir, step, model, model_ema, optimizer)

    indices, losses = compute_loss_per_image(
        model_ema.module if hasattr(model_ema, 'module') else model_ema,
        trainset,
        device
    )
    top_indices = select_top_k_percent(indices, losses, percent=0.7)
    with open('top_val_indices.txt', 'w') as f:
        for idx in top_indices:
            f.write(f"{idx}\n")



    


if __name__ == '__main__':
    main()
