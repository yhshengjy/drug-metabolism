from DeepPurpose import utils, CompoundPred
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

from sklearn.model_selection import RepeatedStratifiedKFold, ParameterGrid

import numpy as np
import pandas as pd
import os
import argparse
import json


def save_json(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(obj, file, indent=4, ensure_ascii=False)


def run_modelt(X1, y, cls_hidden_dims, lr, batch_size, cv, model_name, epoch, train_random_state):

    roc_list = []
    fold_metrics = []

    for i, (train_index, test_index) in enumerate(cv.split(X1, y)):
        print("Fold {}:".format(i))

        X_train = X1.iloc[train_index].tolist()
        y_train = y.iloc[train_index].to_numpy()
        X_val = X1.iloc[test_index].tolist()
        y_val = y.iloc[test_index].to_numpy()

        drug_encoding = model_name

        train = data_process(
            X_drug=X_train,
            y=y_train,
            drug_encoding=drug_encoding,
            split_method='no_split',
            random_seed=train_random_state
        )
        val = data_process(
            X_drug=X_val,
            y=y_val,
            drug_encoding=drug_encoding,
            split_method='no_split',
            random_seed=train_random_state
        )

        config = utils.generate_config(
            drug_encoding=drug_encoding,
            cls_hidden_dims=[cls_hidden_dims],
            train_epoch=epoch,
            LR=lr,
            batch_size=batch_size
        )

        model = CompoundPred.model_initialize(**config)
        model.train(train=train, val=val, test=None, verbose=False)

        scores_tr = model.predict(train)
        scores_va = model.predict(val)

        trm = eval_metric(y_true=train.Label.values, y_prob=scores_tr)
        vam = eval_metric(y_true=val.Label.values, y_prob=scores_va)

        fold_result = {
            'fold': i,
            'train_metrics': trm,
            'val_metrics': vam
        }
        fold_metrics.append(fold_result)

        roc_list.append(float(vam['roc_auc']))
        print(vam)

    mean_train_metrics = {}
    std_train_metrics = {}
    train_metric_keys = fold_metrics[0]['train_metrics'].keys()

    for key in train_metric_keys:
        values = [float(fm['train_metrics'][key]) for fm in fold_metrics]
        mean_train_metrics[key] = float(np.mean(values))
        std_train_metrics[key] = float(np.std(values))

    mean_val_metrics = {}
    std_val_metrics = {}
    val_metric_keys = fold_metrics[0]['val_metrics'].keys()

    for key in val_metric_keys:
        values = [float(fm['val_metrics'][key]) for fm in fold_metrics]
        mean_val_metrics[key] = float(np.mean(values))
        std_val_metrics[key] = float(np.std(values))

    summary = {
        'model_name': model_name,
        'params': {
            'batch_size': batch_size,
            'cls_hidden_dims': cls_hidden_dims,
            'lr': lr
        },
        'mean_train_metrics': mean_train_metrics,
        'std_train_metrics': std_train_metrics,
        'mean_val_metrics': mean_val_metrics,
        'std_val_metrics': std_val_metrics,
        'mean_val_roc_auc': float(np.mean(roc_list)),
        'std_val_roc_auc': float(np.std(roc_list)),
        'num_folds': len(fold_metrics)
    }

    return float(np.mean(roc_list)), fold_metrics, summary


def train_and_evaluate(x, y, model_name, params, epoch, n_splits, n_repeats, cv_random_state, train_random_state):
    batch_size = params['batch_size']
    cls_hidden_dims = params['cls_hidden_dims']
    lr = params['lr']

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=cv_random_state
    )

    rm, fold_metrics, summary = run_modelt(
        X1=x,
        y=y,
        cls_hidden_dims=cls_hidden_dims,
        lr=lr,
        batch_size=batch_size,
        cv=cv,
        model_name=model_name,
        epoch=epoch,
        train_random_state=train_random_state
    )

    return rm, fold_metrics, summary


def train(args):

    t11 = pd.read_csv(args.train_file)

    if 'SMILES' not in t11.columns or 'label' not in t11.columns:
        raise ValueError("file must contain 'SMILES' and 'label' column")

    t11 = t11.dropna(subset=['SMILES', 'label']).copy()

    X1 = t11['SMILES'].apply(process_smiles)
    y = t11['label']

    valid_mask = X1.notna()
    X1 = X1[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    encoding_list = args.encoding_list

    param_grid = {
        'batch_size': args.batch_size,
        'cls_hidden_dims': args.cls_hidden_dims,
        'lr': args.lr
    }
    param_combinations = list(ParameterGrid(param_grid))

    for m in encoding_list:
        print("========== Model: {} ==========".format(m))

        sp = os.path.join(args.tune_save_dir, m)
        try:
            os.makedirs(sp, exist_ok=True)
            print("Directory {} already exists or has been created.".format(sp))
        except Exception as e:
            print("An error occurred while creating the directory: {}".format(e))

        best_roc = -1
        best_params = None
        best_fold_metrics = None
        best_summary = None


        for params in param_combinations:
            print("============== Params: {} ==============".format(params))

            roc, fold_metrics, summary = train_and_evaluate(
                x=X1,
                y=y,
                model_name=m,
                params=params,
                epoch=args.epoch,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                cv_random_state=args.cv_random_state,
                train_random_state=args.train_random_state
            )

            if roc > best_roc:
                best_roc = roc
                best_params = params
                best_fold_metrics = fold_metrics
                best_summary = summary

            print("Current model: {}".format(m))
            print("Current best roc_auc: {}".format(best_roc))


        best_param_info = {
            'model_name': m,
            'best_params': best_params,
            'best_mean_val_roc_auc': float(best_roc),
            'best_std_val_roc_auc': float(best_summary['std_val_roc_auc']),
            'cv_random_state': args.cv_random_state,
            'train_random_state': args.train_random_state
        }
        save_json(
            best_param_info,
            os.path.join(sp, '{}_best_params.json'.format(m))
        )

        save_json(
            best_fold_metrics,
            os.path.join(sp, '{}_best_cv_fold_metrics.json'.format(m))
        )

        save_json(
            best_summary,
            os.path.join(sp, '{}_best_cv_summary.json'.format(m))
        )


        print("==== {} ===== Best CV ROC-AUC: {} ± {}".format(
            m,
            best_summary['mean_val_roc_auc'],
            best_summary['std_val_roc_auc']
        ))
        print("==== {} ===== Best Params: {}".format(m, best_params))
        print("==== {} ===== CV random state: {}".format(m, args.cv_random_state))
        print("==== {} ===== Train random state: {}".format(m, args.train_random_state))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepPurpose compound prediction CV training script')

    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training csv file')

    parser.add_argument('--tune_save_dir', type=str, required=True,
                        help='Directory to save tuning and CV metric results')

    parser.add_argument('--epoch', type=int, required=True,
                        help='Training epochs')

    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds in repeated stratified k-fold')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats in repeated stratified k-fold')

    parser.add_argument('--cv_random_state', type=int, default=1,
                        help='Random seed for CV splitting')

    parser.add_argument('--train_random_state', type=int, default=1,
                        help='Random seed for DeepPurpose data processing / training')

    parser.add_argument('--encoding_list', nargs='+',
                        default=['DGL_AttentiveFP', 'DGL_GCN', 'MPNN', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred'],
                        help='List of drug encoding models')

    parser.add_argument('--batch_size', nargs='+', type=int, default=[32, 64],
                        help='Batch sizes for grid search')
    parser.add_argument('--cls_hidden_dims', nargs='+', type=int, default=[128, 256, 512],
                        help='Single-layer hidden dimensions for grid search')
    parser.add_argument('--lr', nargs='+', type=float, default=[0.0005, 0.001, 0.005, 0.01],
                        help='Learning rates for grid search')

    args = parser.parse_args()
    train(args)