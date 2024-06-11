from tqdm import tqdm
from glob import glob
import argparse
import os
import torch
import numpy as np

from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
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
        subset_str = 'all'
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

    dataset = HuProtDataset(subset=args.subset, classification=True)

    model = ProtBERT(device=device)
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

    loss_fn = sigmoid_focal_loss

    mode_mapper = {
        'val_loss_epoch': 'min',
        'val_acc': 'max',
        'val_auroc': 'max',
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

        num_train_samples = 0
        for iter_idx, (sequence, y_true) in enumerate(train_loader):
            if num_train_samples >= args.max_training_iters:
                break

            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_train_samples += 1

            y_true = y_true.float().to(device)
            y_pred_logit = model(sequence)
            y_pred = torch.sigmoid(y_pred_logit)

            loss = loss_fn(y_pred_logit.flatten(), y_true.flatten())
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
        acc = accuracy_score(y_true=y_true_arr > 0.5, y_pred=y_pred_arr > 0.5)
        auroc = roc_auc_score(y_true=y_true_arr > 0.5, y_score=y_pred_arr)

        log('Train [%s/%s] loss: %.3f, ACC: %.3f, AUROC: %.3f'
            % (epoch_idx + 1, args.n_epochs, train_loss, acc, auroc),
            filepath=args.log_dir,
            to_console=False)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            num_val_samples = 0
            y_true_arr, y_pred_arr = None, None
            for (sequence, y_true) in val_loader:
                sequence = sequence[0]

                # NOTE: sad workaround due to limited GPU memory.
                if len(sequence) > args.max_seq_length:
                    continue
                num_val_samples += 1

                y_true = y_true.float().to(device)
                y_pred_logit = model(sequence)
                y_pred = torch.sigmoid(y_pred_logit)

                loss = loss_fn(y_pred_logit.flatten(), y_true.flatten())
                val_loss += loss.item()

                if y_true_arr is None:
                    y_true_arr = y_true.flatten().detach().cpu().numpy()
                    y_pred_arr = y_pred.flatten().detach().cpu().numpy()
                else:
                    y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                    y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

            val_loss = val_loss / num_val_samples
            acc = accuracy_score(y_true=y_true_arr > 0.5, y_pred=y_pred_arr > 0.5)
            auroc = roc_auc_score(y_true=y_true_arr > 0.5, y_score=y_pred_arr)

            log('Validation [%s/%s] loss: %.3f, ACC: %.3f, AUROC: %.3f'
                % (epoch_idx + 1, args.n_epochs, val_loss, acc, auroc),
                filepath=args.log_dir,
                to_console=False)

            if args.model_saving_metric == 'val_loss_epoch' and val_loss < best_metric:
                best_metric = val_loss
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_acc' and acc > best_metric:
                best_metric = acc
                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), args.model_save_path)
                log('Saving best model (based on %s) to %s.' % (args.model_saving_metric, args.model_save_path),
                    filepath=args.log_dir)
            elif args.model_saving_metric == 'val_auroc' and auroc > best_metric:
                best_metric = auroc
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
        for (sequence, y_true) in test_loader:
            sequence = sequence[0]

            # NOTE: sad workaround due to limited GPU memory.
            if len(sequence) > args.max_seq_length:
                continue
            num_test_samples += 1

            y_true = y_true.float().to(device)
            y_pred_logit = model(sequence)
            y_pred = torch.sigmoid(y_pred_logit)

            loss = loss_fn(y_pred_logit.flatten(), y_true.flatten())
            test_loss += loss.item()

            if y_true_arr is None:
                y_true_arr = y_true.flatten().detach().cpu().numpy()
                y_pred_arr = y_pred.flatten().detach().cpu().numpy()
            else:
                y_true_arr = np.hstack((y_true_arr, y_true.flatten().detach().cpu().numpy()))
                y_pred_arr = np.hstack((y_pred_arr, y_pred.flatten().detach().cpu().numpy()))

        test_loss = test_loss / num_test_samples
        acc = accuracy_score(y_true=y_true_arr > 0.5, y_pred=y_pred_arr > 0.5)
        auroc = roc_auc_score(y_true=y_true_arr > 0.5, y_score=y_pred_arr)

        log('Test loss: %.3f, ACC: %.3f, AUROC: %.3f' % (test_loss, acc, auroc),
            filepath=args.log_dir,
            to_console=False)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('HuProt classification (test set)', fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
    ax.plot(fpr, tpr, label='Ours (AUROC = %.3f)' % (auroc), color='firebrick')

    # Plot the by-chance AUROC.
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='Chance (AUROC = 0.5)', linestyle=':', color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)

    ax.legend(loc='lower right')

    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout(pad=2)
    fig_save_path = args.save_folder + 'HuProt_score_test.png'
    fig.savefig(fig_save_path)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    # data argmuments
    parser.add_argument("--subset", default=None, type=str)

    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-training-iters", default=2048, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--random-seed", default=1, type=int)

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-3, type=float)
    parser.add_argument("--n-epochs", default=100, type=int)
    parser.add_argument("--max-seq-length", default=1024, type=int)  # due to limited GPU memory

    parser.add_argument("--output-save-folder", default='../results/HuProt_classification/', type=str)
    parser.add_argument("--run-count", default=None, type=int)

    # Monitor checkpoint
    parser.add_argument("--model-saving-metric", default='val_auroc', type=str)

    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    assert cl_args.run_count is None
    cl_args = parse_settings(cl_args)
    main(cl_args)

