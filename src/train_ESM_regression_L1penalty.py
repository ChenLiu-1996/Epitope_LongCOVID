from tqdm import tqdm
from glob import glob
import argparse
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from contextlib import nullcontext

from dataset.HuProt import HuProtDataset
from nn.esm import ESM
from utils.log import log
from utils.seed import seed_everything
from utils.split import split_dataset
from utils.scheduler import LinearWarmupCosineAnnealingLR


def attention_rollout(attentions, discard_ratio=0.95, head_fusion='max', use_grad=False):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    context = nullcontext() if use_grad is True else torch.no_grad()
    with context:
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but don't drop the class token.
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attentions[0].device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    return result

def soft_topk(x, k, temperature=1.0, dim=-1):
    weights = torch.softmax(x / temperature, dim=-1)
    return torch.topk(weights, k, dim=dim).values

def parse_settings(args):
    # Initialize save folder.
    if args.subset is None:
        subset_str = 'all'
    else:
        subset_str = args.subset

    setting_str = 'lr-%s_L1penalty-%s_iter-%s_layer-%s_seed_%d' % (
        args.lr, args.L1_coeff, args.max_training_iters, args.num_esm_layers, args.random_seed
    )

    output_save_path = os.path.join(args.output_save_folder, subset_str, setting_str)

    existing_runs = glob(os.path.join(output_save_path, 'run_*/'))
    if len(existing_runs) > 0:
        run_counts = [int(item.split('/')[-2].split('run_')[1]) for item in existing_runs]
        run_count = max(run_counts) + 1
    else:
        run_count = 1

    args.save_folder = os.path.join(output_save_path, 'run_%d/' % run_count)
    args.model_save_path = os.path.join(args.save_folder, 'model_best_%s.pty' % args.model_saving_metric)

    # Initialize log file.
    args.log_dir = os.path.join(args.save_folder, 'log.txt')

    log_str = 'Config: \n'
    print(args)
    for key in args.__dict__:
        log_str += '%s: %s\n' % (key, getattr(args, key))
    log_str += '\nTraining History:'
    log(log_str, filepath=args.log_dir, to_console=True)

    return args


def main(args):
    seed_everything(args.random_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    dataset = HuProtDataset(subset=args.subset, classification=False)

    model = ESM(device=device,
                num_esm_layers=args.num_esm_layers,
                )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.n_epochs//4,
                                              warmup_start_lr=args.lr/1000,
                                              max_epochs=args.n_epochs)

    train_set, val_set, test_set = split_dataset(dataset, splits=(0.8, 0.1, 0.1), random_seed=1)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    loss_fn_pred = torch.nn.MSELoss()

    mode_mapper = {
        'val_loss_epoch': 'min',
        'val_pearson_R': 'max',
        'val_spearman_R': 'max',
        'val_recon_acc': 'max',
    }

    if mode_mapper[args.model_saving_metric] == 'min':
        best_metric = np.inf
    elif mode_mapper[args.model_saving_metric] == 'max':
        best_metric = -np.inf

    # NOTE: Training.
    for epoch_idx in tqdm(range(args.n_epochs)):
        train_loss = 0
        y_true_arr, y_pred_arr = None, None
        optimizer.zero_grad()
        model.train()

        num_train_samples = 0
        for iter_idx, (sequence, y_true) in enumerate(train_loader):
            y_true = y_true.float().to(device)

            if num_train_samples >= args.max_training_iters:
                break

            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_train_samples += 1

            y_pred, attentions = model.output_attentions(sequence)
            loss = loss_fn_pred(y_pred.flatten(), y_true.flatten())

            # Additional L1 penalty.
            attention_rolled = attention_rollout(attentions, use_grad=True)
            final_attribution = torch.sum(attention_rolled, dim=-1)
            # Drop START and END tokens.
            final_attribution = final_attribution[1:-1]
            L1_penalty = final_attribution.abs().mean()
            loss += args.L1_coeff * L1_penalty
            train_loss += loss.item()

            # Train loss aggregation to simulate the target batch size.
            loss = loss / args.batch_size
            loss.backward()
            if iter_idx % args.batch_size == (args.batch_size - 1):
                # # gradient clipping to avoid gradient explosion
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

        train_loss = train_loss / num_train_samples
        pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
        spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
        scheduler.step()

        log('Train [%s/%s] loss (recon): %.3f, P.R: %.3f, S.R: %.3f'
            % (epoch_idx + 1, args.n_epochs, train_loss, pearson_R, spearman_R),
            filepath=args.log_dir,
            to_console=False)

        # NOTE: Validation.
        with torch.no_grad():
            model.eval()
            val_loss = 0
            num_val_samples = 0
            y_true_arr, y_pred_arr = None, None

            for (sequence, y_true) in val_loader:
                y_true = y_true.float().to(device)

                sequence = sequence[0]

                # NOTE: sad workaround due to limited GPU memory.
                if len(sequence) > args.max_seq_length:
                    continue
                num_val_samples += 1

                y_pred, attentions = model.output_attentions(sequence)
                loss = loss_fn_pred(y_pred.flatten(), y_true.flatten())
                # Additional L1 penalty.
                attention_rolled = attention_rollout(attentions)
                final_attribution = torch.sum(attention_rolled, dim=-1)
                # Drop START and END tokens.
                final_attribution = final_attribution[1:-1]
                L1_penalty = final_attribution.abs().mean()
                loss += args.L1_coeff * L1_penalty
                val_loss += loss.item()

                if y_true_arr is None:
                    y_true_arr = y_true.flatten().detach().cpu().numpy()
                    y_pred_arr = y_pred.flatten().detach().cpu().numpy()
                else:
                    y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                    y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

            val_loss = val_loss / num_val_samples
            pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
            spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]

            log('Validation [%s/%s] loss (recon): %.3f, P.R: %.3f, S.R: %.3f'
                % (epoch_idx + 1, args.n_epochs, val_loss, pearson_R, spearman_R),
                filepath=args.log_dir,
                to_console=False)

            if args.model_saving_metric == 'val_loss_epoch' and val_loss < best_metric:
                best_metric = val_loss
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_pearson_R' and pearson_R > best_metric:
                best_metric = pearson_R
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_spearman_R' and spearman_R > best_metric:
                best_metric = spearman_R
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)

    # NOTE: Testing.
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_loss = 0
        num_test_samples = 0
        y_true_arr, y_pred_arr = None, None
        sequence_list, attribution_list = [], []

        for (sequence, y_true) in test_loader:
            y_true = y_true.float().to(device)
            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_test_samples += 1

            y_pred, attentions = model.output_attentions(sequence)
            attentions = [item.cpu() for item in attentions]
            attention_rolled = attention_rollout(attentions)
            final_attribution = torch.sum(attention_rolled, dim=-1)

            batch_size = final_attribution.shape[0]
            assert batch_size == 1
            final_attribution = final_attribution.flatten().detach().numpy().tolist()
            final_attribution = [np.round(item, 4) for item in final_attribution]
            # Drop START and END tokens.
            final_attribution = final_attribution[1:-1]
            assert len(final_attribution) == len(sequence)

            sequence_list.append(sequence)
            attribution_list.append(final_attribution)

            loss = loss_fn_pred(y_pred.flatten(), y_true.flatten())
            # Additional L1 penalty.
            L1_penalty = final_attribution.abs().mean()
            loss += args.L1_coeff * L1_penalty
            test_loss += loss.item()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

        test_loss = test_loss / num_test_samples
        pearson_R, pearson_P = pearsonr(y_true_arr, y_pred_arr)
        spearman_R, spearman_P = spearmanr(a=y_true_arr, b=y_pred_arr)

        log('Test loss (recon): %.3f, P.R: %.3f, S.R: %.3f' % (
            test_loss, pearson_R, spearman_R),
            filepath=args.log_dir,
            to_console=False)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('HuProt score prediction (test set)', fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(y_true_arr, y_pred_arr,
               marker='o', facecolors='skyblue', edgecolors='black', alpha=0.5, s=80)
    # Best Line Fit.
    coefficients = np.polyfit(y_true_arr, y_pred_arr, 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(y_true_arr.min(), y_true_arr.max(), 1000)
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit, color='black', linestyle=':')
    ax.set_xlabel('Ground Truth HuProt scores', fontsize=18)
    ax.set_ylabel('Predicted HuProt scores', fontsize=18)
    ax.set_title('Pearson R = %.3f (p = %.3f), Spearman R = %.3f (p = %.3f)' % (
        pearson_R, pearson_P, spearman_R, spearman_P
    ), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout(pad=2)
    fig_save_path = os.path.join(args.save_folder, 'HuProt_score_test.png')
    fig.savefig(fig_save_path)
    plt.close(fig)

    # NOTE: plotting the attention attributions.
    # Top k highest true HuProt score.
    # Top k lowest true HuProt score.
    # Top k highest predicted HuProt score.
    # Top k lowest predicted HuProt score.
    topk = 10

    fig = plt.figure(figsize=(30, 15))
    gs = plt.GridSpec(2 * topk + 1, 2, width_ratios=[95, 5])
    indices = np.argsort(y_true_arr)[-topk:][::-1]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[row_idx, 0])
        cbar_data = ax.imshow([attribution_list[item_idx]], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score: {y_true_arr[item_idx]:.2f}, Pred HuProt score: {y_pred_arr[item_idx]:.2f}')
    indices = np.argsort(y_true_arr)[:topk]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[topk + 1 + row_idx, 0])
        cbar_data = ax.imshow([attribution_list[item_idx]], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score: {y_true_arr[item_idx]:.2f}, Pred HuProt score: {y_pred_arr[item_idx]:.2f}')
    ax_colorbar = fig.add_subplot(gs[:, 1])
    fig.colorbar(cbar_data, ax=ax_colorbar, orientation='vertical', aspect=90)
    ax_colorbar.set_axis_off()
    fig_save_path = os.path.join(args.save_folder, 'Attention_Attribution_topk_true_HuProt_score.png')
    fig.tight_layout(pad=2)
    fig.savefig(fig_save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(30, 15))
    gs = plt.GridSpec(2 * topk + 1, 2, width_ratios=[95, 5])
    indices = np.argsort(y_pred_arr)[-topk:][::-1]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[row_idx, 0])
        cbar_data = ax.imshow([attribution_list[item_idx]], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score: {y_true_arr[item_idx]:.2f}, Pred HuProt score: {y_pred_arr[item_idx]:.2f}')
    indices = np.argsort(y_pred_arr)[:topk]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[topk + 1 + row_idx, 0])
        cbar_data = ax.imshow([attribution_list[item_idx]], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score: {y_true_arr[item_idx]:.2f}, Pred HuProt score: {y_pred_arr[item_idx]:.2f}')
    ax_colorbar = fig.add_subplot(gs[:, 1])
    fig.colorbar(cbar_data, ax=ax_colorbar, orientation='vertical', aspect=90)
    ax_colorbar.set_axis_off()
    fig_save_path = os.path.join(args.save_folder, 'Attention_Attribution_topk_pred_HuProt_score.png')
    fig.tight_layout(pad=2)
    fig.savefig(fig_save_path)
    plt.close(fig)

    # Save the data.
    df = pd.DataFrame({
        'Sequence': sequence_list,
        'Attribution': attribution_list,
        'HuProt_true': y_true_arr,
        'HuProt_pred': y_pred_arr,
    })

    df.to_csv(os.path.join(args.save_folder, 'data.csv'), index=False)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    # data argmuments
    parser.add_argument("--subset", default=None, type=str)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-training-iters", default=4096, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--num-esm-layers", default=None, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float)
    parser.add_argument("--L1-coeff", default=1e-1, type=float)
    parser.add_argument("--n-epochs", default=50, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)  # due to limited GPU memory

    parser.add_argument("--output-save-folder", default='../results/ESM_HuProt_regression_L1penalty/', type=str)
    parser.add_argument("--run-count", default=None, type=int)

    # Monitor checkpoint
    parser.add_argument("--model-saving-metric", default='val_loss_epoch', type=str)

    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    assert cl_args.run_count is None
    cl_args = parse_settings(cl_args)
    main(cl_args)

