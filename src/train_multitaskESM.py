from tqdm import tqdm
from glob import glob
import argparse
import os
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import phate
import scprep

from dataset.HuProt import HuProtDataset
from nn.recon_esm import ReconESM
from utils.log import log
from utils.seed import seed_everything
from utils.split import split_dataset


def parse_settings(args):
    # Initialize save folder.
    if args.subset is None:
        subset_str = 'all'
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

    model = ReconESM(device=device,
                     num_esm_layers=args.num_esm_layers,
                     mask_ratio=args.mask_ratio,
                     max_seq_len=args.max_seq_length * 2)  # Space-separated sequence, hence 2x length.
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.wd)

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
    loss_fn_recon = torch.nn.CrossEntropyLoss()

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
        num_token, num_token_correct = 0, 0
        optimizer.zero_grad()
        model.train()

        num_train_samples = 0
        for iter_idx, (sequence, y_true) in enumerate(train_loader):
            y_true = torch.log10(y_true.float().to(device))

            if num_train_samples >= args.max_training_iters:
                break

            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_train_samples += 1

            y_pred, seq_input, seq_recon_logit, hidden_state = model(sequence)

            loss_reg = loss_fn_pred(y_pred.flatten(), y_true.flatten())
            loss_recon = loss_fn_recon(seq_recon_logit, seq_input)
            loss = loss_reg + args.coeff_recon * loss_recon

            train_loss += loss.item()

            # Train loss aggregation to simulate the target batch size.
            loss = loss / args.batch_size
            loss.backward()
            if iter_idx % args.batch_size == (args.batch_size - 1):
                optimizer.step()
                optimizer.zero_grad()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

            token_true = seq_input.flatten()
            token_recon = seq_recon_logit.argmax(1).flatten()
            num_token += len(token_true)
            num_token_correct += (token_true == token_recon).sum()

        train_loss = train_loss / num_train_samples
        pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
        spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
        token_recon_acc = num_token_correct / num_token * 100

        log('Train [%s/%s] loss (recon): %.3f, P.R: %.3f, S.R: %.3f, token_recon_acc: %.2f'
            % (epoch_idx + 1, args.n_epochs, train_loss, pearson_R, spearman_R, token_recon_acc),
            filepath=args.log_dir,
            to_console=False)

        # NOTE: Validation.
        with torch.no_grad():
            model.eval()
            val_loss = 0
            num_val_samples = 0
            y_true_arr, y_pred_arr = None, None
            num_token, num_token_correct = 0, 0

            for (sequence, y_true) in val_loader:
                y_true = torch.log10(y_true.float().to(device))

                sequence = sequence[0]

                # NOTE: sad workaround due to limited GPU memory.
                if len(sequence) > args.max_seq_length:
                    continue
                num_val_samples += 1

                y_pred, seq_input, seq_recon_logit, hidden_state = model(sequence)

                loss_reg = loss_fn_pred(y_pred.flatten(), y_true.flatten())
                loss_recon = loss_fn_recon(seq_recon_logit, seq_input)
                loss = loss_reg + args.coeff_recon * loss_recon

                val_loss += loss.item()

                if y_true_arr is None:
                    y_true_arr = y_true.flatten().detach().cpu().numpy()
                    y_pred_arr = y_pred.flatten().detach().cpu().numpy()
                else:
                    y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                    y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

                token_true = seq_input.flatten()
                token_recon = seq_recon_logit.argmax(1).flatten()
                num_token += len(token_true)
                num_token_correct += (token_true == token_recon).sum()

            val_loss = val_loss / num_val_samples
            pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
            spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
            token_recon_acc = num_token_correct / num_token * 100

            log('Validation [%s/%s] loss (recon): %.3f, P.R: %.3f, S.R: %.3f, token_recon_acc: %.2f'
                % (epoch_idx + 1, args.n_epochs, val_loss, pearson_R, spearman_R, token_recon_acc),
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
            elif args.model_saving_metric == 'val_recon_acc' and token_recon_acc > best_metric:
                best_metric = token_recon_acc
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
        y_true_arr, y_pred_arr, hidden_state_arr = None, None, None
        num_token, num_token_correct = 0, 0

        for (sequence, y_true) in test_loader:
            y_true = torch.log10(y_true.float().to(device))
            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_test_samples += 1

            y_pred, seq_input, seq_recon_logit, hidden_state = model(sequence)

            loss_reg = loss_fn_pred(y_pred.flatten(), y_true.flatten())
            loss_recon = loss_fn_recon(seq_recon_logit, seq_input)
            loss = loss_reg + args.coeff_recon * loss_recon

            test_loss += loss.item()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
                hidden_state_arr = torch.mean(hidden_state, dim=1, keepdim=True).squeeze(1).detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))
                hidden_state_arr = np.vstack((hidden_state_arr, torch.mean(hidden_state, dim=1, keepdim=True).squeeze(1).detach().cpu().numpy()))

            token_true = seq_input.flatten()
            token_recon = seq_recon_logit.argmax(1).flatten()
            num_token += len(token_true)
            num_token_correct += (token_true == token_recon).sum()

        test_loss = test_loss / num_test_samples
        pearson_R, pearson_P = pearsonr(y_true_arr, y_pred_arr)
        spearman_R, spearman_P = spearmanr(a=y_true_arr, b=y_pred_arr)
        token_recon_acc = num_token_correct / num_token * 100

        log('Test loss (recon): %.3f, P.R: %.3f, S.R: %.3f, token_recon_acc: %.2f' % (
            test_loss, pearson_R, spearman_R, token_recon_acc),
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
    ax.set_xlabel('log10( Ground Truth HuProt scores )', fontsize=18)
    ax.set_ylabel('log10( Predicted HuProt scores )', fontsize=18)
    ax.set_title('Pearson R = %.3f (p = %.3f), Spearman R = %.3f (p = %.3f)' % (
        pearson_R, pearson_P, spearman_R, spearman_P
    ), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout(pad=2)
    fig_save_path = os.path.join(args.save_folder, 'HuProt_score_test.png')
    fig.savefig(fig_save_path)
    plt.close(fig)

    # Plot the latent space.
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('HuProt latent space', fontsize=20)
    phate_op = phate.PHATE(random_state=1, n_jobs=1)
    data_phate = phate_op.fit_transform(hidden_state_arr)

    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    scprep.plot.scatter2d(
        data_phate,
        c=y_true_arr,
        ax=ax,
        title='',
        xticks=False,
        yticks=False,
        colorbar=True,
        label_prefix='PHATE',
        fontsize=10,
        s=3)

    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout(pad=2)
    fig_save_path = os.path.join(args.save_folder, 'latent_space.png')
    fig.savefig(fig_save_path)
    plt.close(fig)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    # data argmuments
    parser.add_argument("--subset", default=None, type=str)

    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-training-iters", default=2048, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--num-esm-layers", default=None, type=int)

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-3, type=float)
    parser.add_argument("--n-epochs", default=50, type=int)
    parser.add_argument("--max-seq-length", default=1024, type=int)  # due to limited GPU memory
    parser.add_argument("--mask-ratio", default=0.2, type=float)
    parser.add_argument("--coeff-recon", default=1e-1, type=float)

    parser.add_argument("--output-save-folder", default='../results/MultitaskESM_HuProt/', type=str)
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

