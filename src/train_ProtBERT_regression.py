from tqdm import tqdm
from glob import glob
import argparse
import os
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset.HuProt import HuProtDataset
from nn.protbert import ProtBERT
from utils.log import log
from utils.seed import seed_everything
from utils.split import split_dataset


def parse_settings(args):
    # Initialize save folder.
    if args.subset is None:
        subset_str = 'All'
    else:
        subset_str = args.subset
    output_save_path = args.output_save_folder + '/' + subset_str

    existing_runs = glob(output_save_path + '/run_*/')
    if len(existing_runs) > 0:
        run_counts = [int(item.split('/')[-2].split('run_')[1]) for item in existing_runs]
        run_count = max(run_counts) + 1
    else:
        run_count = 1

    args.save_folder = '%s/run_%d/' % (output_save_path, run_count)
    args.model_save_path = args.save_folder + '/model_best_%s.pty' % args.model_saving_metric

    # Initialize log file.
    args.log_dir = args.save_folder + 'log.txt'

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

    model = ProtBERT(device=device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.wd)

    train_set, val_set, test_set = split_dataset(dataset, splits=(0.6, 0.2, 0.2), random_seed=1)

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

    loss_fn = torch.nn.MSELoss()

    mode_mapper = {
        'val_loss_epoch': 'min',
        'val_pearson_R': 'max',
        'val_spearman_R': 'max',
    }

    if mode_mapper[args.model_saving_metric] == 'min':
        best_metric = np.inf
    elif mode_mapper[args.model_saving_metric] == 'max':
        best_metric = -np.inf

    #
    for epoch_idx in tqdm(range(args.n_epochs)):
        model.train()

        train_loss = 0
        y_true_arr, y_pred_arr = None, None
        optimizer.zero_grad()
        model.train()
        for batch_idx, (symbol, sequence, y_true) in enumerate(train_loader):
            sequence = sequence[0]

            y_true = y_true.float().to(device)
            y_pred = model(sequence)

            loss = loss_fn(y_pred.flatten(), y_true.flatten())
            train_loss += loss.item()

            # Train loss aggregation to simulate the target batch size.
            loss = loss / args.batch_size
            loss.backward()
            if batch_idx % args.batch_size == (args.batch_size - 1):
                optimizer.step()
                optimizer.zero_grad()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

        train_loss = train_loss / len(train_loader.dataset)
        pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
        spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]

        log('Train [%s/%s] loss: %.3f, P.R: %.3f, S.R: %.3f'
            % (epoch_idx + 1, args.n_epochs, train_loss, pearson_R, spearman_R),
            filepath=args.log_dir,
            to_console=False)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            y_true_arr, y_pred_arr = None, None
            for (symbol, sequence, y_true) in val_loader:
                sequence = sequence[0]

                y_true = y_true.float().to(device)
                y_pred = model(sequence)

                loss = loss_fn(y_pred.flatten(), y_true.flatten())
                val_loss += loss.item()

                if y_true_arr is None:
                    y_true_arr = y_true.flatten().detach().cpu().numpy()
                    y_pred_arr = y_pred.flatten().detach().cpu().numpy()
                else:
                    y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                    y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

            val_loss = val_loss / len(val_loader.dataset)
            pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
            spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]

            log('Validation [%s/%s] loss: %.3f, P.R: %.3f, S.R: %.3f'
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
        y_true_arr, y_pred_arr = None, None
        for (symbol, sequence, y_true) in test_loader:
            sequence = sequence[0]

            y_true = y_true.float().to(device)
            y_pred = model(sequence)

            loss = loss_fn(y_pred.flatten(), y_true.flatten())
            test_loss += loss.item()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

        test_loss = test_loss / len(test_loader.dataset)
        pearson_R, pearson_P = pearsonr(y_true_arr, y_pred_arr)
        spearman_R, spearman_P = spearmanr(a=y_true_arr, b=y_pred_arr)

        log('Test loss: %.3f, P.R: %.3f, S.R: %.3f' % (test_loss, pearson_R, spearman_R),
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
    fig_save_path = args.save_folder + 'HuProt_score_test.png'
    fig.savefig(fig_save_path)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    # data argmuments
    parser.add_argument("--subset", default=None, type=str)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--random-seed", default=1, type=int)

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-3, type=float)
    parser.add_argument("--n-epochs", default=200, type=int)

    parser.add_argument("--output-save-folder", default='../results/HuProt_regression/', type=str)
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

