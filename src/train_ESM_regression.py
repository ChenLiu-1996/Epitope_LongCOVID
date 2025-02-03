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

from dataset.HuProt import HuProtDataset
from nn.esm import ESM
from utils.log import log
from utils.seed import seed_everything
from utils.split import split_dataset
from utils.scheduler import LinearWarmupCosineAnnealingLR



def parse_settings(args):
    # Initialize save folder.
    if args.subset is None:
        subset_str = '3way'
    else:
        subset_str = args.subset

    setting_str = 'lr-%s_iter-%s_layer-%s_seed_%d' % (
        args.lr, args.max_training_iters, args.num_esm_layers, args.random_seed
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
                num_esm_layers=args.num_esm_layers)
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

            y_pred = model(sequence)
            loss = loss_fn_pred(y_pred, y_true)
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
                y_true_arr = y_true.detach().cpu().numpy()
                y_pred_arr = y_pred.detach().cpu().numpy()
            else:
                y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
                y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

        train_loss = train_loss / num_train_samples
        scheduler.step()

        pearson_Rs, spearman_Rs = np.zeros((3,)), np.zeros((3,))
        pearson_Rs[0] = pearsonr(y_true_arr[:, 0], y_pred_arr[:, 0])[0]
        spearman_Rs[0] = spearmanr(a=y_true_arr[:, 0], b=y_pred_arr[:, 0])[0]
        pearson_Rs[1] = pearsonr(y_true_arr[:, 1], y_pred_arr[:, 1])[0]
        spearman_Rs[1] = spearmanr(a=y_true_arr[:, 1], b=y_pred_arr[:, 1])[0]
        pearson_Rs[2] = pearsonr(y_true_arr[:, 2], y_pred_arr[:, 2])[0]
        spearman_Rs[2] = spearmanr(a=y_true_arr[:, 2], b=y_pred_arr[:, 2])[0]

        log('Train [%s/%s] loss: %.3f, P.R|S.R (LC): %.3f|%.3f, P.R|S.R (HC): %.3f|%.3f, P.R|S.R (CVC): %.3f|%.3f'
            % (epoch_idx + 1, args.n_epochs, train_loss, pearson_Rs[0], spearman_Rs[0], pearson_Rs[1], spearman_Rs[1], pearson_Rs[2], spearman_Rs[2]),
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

                y_pred = model(sequence)
                loss = loss_fn_pred(y_pred, y_true)
                val_loss += loss.item()

                if y_true_arr is None:
                    y_true_arr = y_true.detach().cpu().numpy()
                    y_pred_arr = y_pred.detach().cpu().numpy()
                else:
                    y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
                    y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

            val_loss = val_loss / num_val_samples

            pearson_Rs, spearman_Rs = np.zeros((3,)), np.zeros((3,))
            pearson_Rs[0] = pearsonr(y_true_arr[:, 0], y_pred_arr[:, 0])[0]
            spearman_Rs[0] = spearmanr(a=y_true_arr[:, 0], b=y_pred_arr[:, 0])[0]
            pearson_Rs[1] = pearsonr(y_true_arr[:, 1], y_pred_arr[:, 1])[0]
            spearman_Rs[1] = spearmanr(a=y_true_arr[:, 1], b=y_pred_arr[:, 1])[0]
            pearson_Rs[2] = pearsonr(y_true_arr[:, 2], y_pred_arr[:, 2])[0]
            spearman_Rs[2] = spearmanr(a=y_true_arr[:, 2], b=y_pred_arr[:, 2])[0]

            log('Validation [%s/%s] loss: %.3f, P.R|S.R (LC): %.3f|%.3f, P.R|S.R (HC): %.3f|%.3f, P.R|S.R (CVC): %.3f|%.3f'
                % (epoch_idx + 1, args.n_epochs, val_loss, pearson_Rs[0], spearman_Rs[0], pearson_Rs[1], spearman_Rs[1], pearson_Rs[2], spearman_Rs[2]),
                filepath=args.log_dir,
                to_console=False)

            if args.model_saving_metric == 'val_loss_epoch' \
            and val_loss < best_metric:
                best_metric = val_loss
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_pearson_R' \
            and pearson_Rs.mean() > best_metric:
                best_metric = pearson_Rs.mean()
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_spearman_R' \
            and pearson_Rs.mean() > best_metric:
                best_metric = pearson_Rs.mean()
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)

    # NOTE: Testing.
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    test_loss = 0
    num_test_samples = 0
    y_true_arr, y_pred_arr = None, None
    sequence_list, attribution_list_LC, attribution_list_HC, attribution_list_CVC = [], [], [], []

    for (sequence, y_true) in test_loader:
        y_true = y_true.float().to(device)
        sequence = sequence[0]

        # NOTE: sad workaround due to limited GPU memory.
        if len(sequence) > args.max_seq_length:
            continue
        num_test_samples += 1

        token_attribution = model.output_attribution(sequence)
        assert model.num_classes == token_attribution.shape[0]
        # Drop START and END tokens.
        token_attribution = token_attribution[:, 1:-1]
        assert len(sequence) == token_attribution.shape[1]

        sequence_list.append(sequence)
        attribution_list_LC.append([np.round(item, 4) for item in token_attribution[0, :]])
        attribution_list_HC.append([np.round(item, 4) for item in token_attribution[1, :]])
        attribution_list_CVC.append([np.round(item, 4) for item in token_attribution[2, :]])

        loss = loss_fn_pred(y_pred, y_true)
        test_loss += loss.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    test_loss = test_loss / num_test_samples

    pearson_Rs, spearman_Rs = np.zeros((3,)), np.zeros((3,))
    pearson_Ps, spearman_Ps = np.zeros((3,)), np.zeros((3,))
    pearson_Rs[0], pearson_Ps[0] = pearsonr(y_true_arr[:, 0], y_pred_arr[:, 0])
    spearman_Rs[0], spearman_Ps[0] = spearmanr(a=y_true_arr[:, 0], b=y_pred_arr[:, 0])
    pearson_Rs[1], pearson_Ps[1] = pearsonr(y_true_arr[:, 1], y_pred_arr[:, 1])
    spearman_Rs[1], spearman_Ps[1] = spearmanr(a=y_true_arr[:, 1], b=y_pred_arr[:, 1])
    pearson_Rs[2], pearson_Ps[2] = pearsonr(y_true_arr[:, 2], y_pred_arr[:, 2])
    spearman_Rs[2], spearman_Ps[2] = spearmanr(a=y_true_arr[:, 2], b=y_pred_arr[:, 2])

    log('Test loss: %.3f, P.R|S.R (LC): %.3f|%.3f, P.R|S.R (HC): %.3f|%.3f, P.R|S.R (CVC): %.3f|%.3f' % (
        test_loss, pearson_Rs[0], spearman_Rs[0], pearson_Rs[1], spearman_Rs[1], pearson_Rs[2], spearman_Rs[2]),
        filepath=args.log_dir,
        to_console=False)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(26, 8))
    fig.suptitle('HuProt score prediction (test set)', fontsize=20)
    for category_idx, category_name in enumerate(['LC', 'HC', 'CVC']):
        ax = fig.add_subplot(1, 3, category_idx + 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.scatter(y_true_arr[:, category_idx], y_pred_arr[:, category_idx],
                marker='o', facecolors='skyblue', edgecolors='black', alpha=0.5, s=80)
        # Best Line Fit.
        coefficients = np.polyfit(y_true_arr[:, category_idx], y_pred_arr[:, category_idx], 1)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(y_true_arr.min(), y_true_arr.max(), 1000)
        y_fit = polynomial(x_fit)
        ax.plot(x_fit, y_fit, color='black', linestyle=':')
        ax.set_xlabel(f'Ground Truth HuProt scores ({category_name})', fontsize=18)
        ax.set_ylabel(f'Predicted HuProt scores ({category_name})', fontsize=18)
        ax.set_title('Pearson R = %.3f (p = %.3f), Spearman R = %.3f (p = %.3f)' % (
            pearson_Rs[category_idx], pearson_Ps[category_idx], spearman_Rs[category_idx], spearman_Ps[category_idx]
        ), fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout(pad=2)
    fig_save_path = os.path.join(args.save_folder, 'HuProt_score_test.png')
    fig.savefig(fig_save_path)
    plt.close(fig)

    # NOTE: plotting the attention attributions.
    # Top k highest true LC HuProt score.
    # Top k lowest true LC HuProt score.
    # Top k highest predicted LC HuProt score.
    # Top k lowest predicted LC HuProt score.
    topk = 5

    fig = plt.figure(figsize=(30, 15))
    gs = plt.GridSpec(4 * topk + 1, 2, width_ratios=[95, 5])
    indices = np.argsort(y_true_arr[:, 0])[-topk:][::-1]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[row_idx * 2, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_HC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score (LC|HC|CVC): {y_true_arr[item_idx, 0]:.2f}|{y_true_arr[item_idx, 1]:.2f}|{y_true_arr[item_idx, 2]:.2f}, ' + \
                     f'Pred HuProt score (LC|HC|CVC): {y_pred_arr[item_idx, 0]:.2f}|{y_pred_arr[item_idx, 1]:.2f}|{y_pred_arr[item_idx, 2]:.2f}' + \
                     '\nLC - HC')
        ax = fig.add_subplot(gs[row_idx * 2 + 1, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_CVC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title('LC - CVC')

    indices = np.argsort(y_true_arr[:, 0])[:topk]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[topk * 2 + 1 + row_idx * 2, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_HC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score (LC|HC|CVC): {y_true_arr[item_idx, 0]:.2f}|{y_true_arr[item_idx, 1]:.2f}|{y_true_arr[item_idx, 2]:.2f}, ' + \
                     f'Pred HuProt score (LC|HC|CVC): {y_pred_arr[item_idx, 0]:.2f}|{y_pred_arr[item_idx, 1]:.2f}|{y_pred_arr[item_idx, 2]:.2f}' + \
                     '\nLC - HC')

        ax = fig.add_subplot(gs[topk * 2 + 1 + row_idx * 2 + 1, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_CVC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title('\nLC - CVC')

    ax_colorbar = fig.add_subplot(gs[:, 1])
    fig.colorbar(cbar_data, ax=ax_colorbar, orientation='vertical', aspect=90)
    ax_colorbar.set_axis_off()
    fig_save_path = os.path.join(args.save_folder, 'Attention_Attribution_topk_true_HuProt_score.png')
    fig.tight_layout(pad=2)
    fig.savefig(fig_save_path)
    plt.close(fig)

    fig = plt.figure(figsize=(30, 15))
    gs = plt.GridSpec(4 * topk + 1, 2, width_ratios=[95, 5])
    indices = np.argsort(y_pred_arr[:, 0])[-topk:][::-1]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[row_idx * 2, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_HC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score (LC|HC|CVC): {y_true_arr[item_idx, 0]:.2f}|{y_true_arr[item_idx, 1]:.2f}|{y_true_arr[item_idx, 2]:.2f}, ' + \
                     f'Pred HuProt score (LC|HC|CVC): {y_pred_arr[item_idx, 0]:.2f}|{y_pred_arr[item_idx, 1]:.2f}|{y_pred_arr[item_idx, 2]:.2f}' + \
                     '\nLC - HC')
        ax = fig.add_subplot(gs[row_idx * 2 + 1, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_CVC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title('LC - CVC')

    indices = np.argsort(y_pred_arr[:, 0])[:topk]
    for row_idx, item_idx in enumerate(indices):
        ax = fig.add_subplot(gs[topk * 2 + 1 + row_idx * 2, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_HC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title(f'True HuProt score (LC|HC|CVC): {y_true_arr[item_idx, 0]:.2f}|{y_true_arr[item_idx, 1]:.2f}|{y_true_arr[item_idx, 2]:.2f}, ' + \
                     f'Pred HuProt score (LC|HC|CVC): {y_pred_arr[item_idx, 0]:.2f}|{y_pred_arr[item_idx, 1]:.2f}|{y_pred_arr[item_idx, 2]:.2f}' + \
                     '\nLC - HC')

        ax = fig.add_subplot(gs[topk * 2 + 1 + row_idx * 2 + 1, 0])
        cbar_data = ax.imshow([np.array(attribution_list_LC[item_idx]) - np.array(attribution_list_CVC[item_idx])], cmap='inferno', clim=[0, 2.0])
        ax.set_yticks([])
        ax.set_xticks(ticks=np.arange(len(sequence_list[item_idx])), labels=list(sequence_list[item_idx]), fontsize=6, rotation=0)
        ax.set_title('\nLC - CVC')

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
        'Attribution_LC': attribution_list_LC,
        'Attribution_HC': attribution_list_HC,
        'Attribution_CVC': attribution_list_CVC,
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
    parser.add_argument("--n-epochs", default=50, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)  # due to limited GPU memory

    parser.add_argument("--output-save-folder", default='../results/ESM_HuProt_regression/', type=str)
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

