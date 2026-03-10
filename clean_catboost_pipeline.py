
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


# Predictor lists
predictors_0 = ['KJONN']
predictors_1 = ['KJONN', 'F_parent']
predictors_2 = ['KJONN', 'F_parent', 'age_mother', 'age_father', 'ses', 'civil_status', 'smoke',
                'alcohol', 'alcohol_binge', 'coffee_binge', 'folate', 'no_suppl', 'bmi_start',
                'weight_gain', 'diabet', 'blood', 'thyroid', 'fever', 'immune', 'infection',
                'asthma', 'b12', 'anemia', 'apgar5', 'lbw', 'epilepsy', 'oxygen', 'prematur']
predictors_3 = predictors_2 + ['dev_delay_6m', 'sleep', 'wake']
predictors_4 = predictors_2 + ['dev_delay_18m', 'sleep_hours', 'sleep_wakes']
predictors_5 = predictors_2 + ['lang_36', 'dev_delay_36m', 'soc_hesitant', 'mood']
predictors_6 = predictors_2 + ['dev_delay_6m', 'sleep', 'wake', 'dev_delay_18m',
                               'sleep_hours', 'sleep_wakes', 'lang_36', 'dev_delay_36m',
                               'soc_hesitant', 'mood']
predictors_7 = predictors_5 + ['ASD.prs.pc']
predictors_8 = predictors_6 + ['ASD.prs.pc']

predictors_list = [predictors_0, predictors_1, predictors_2, predictors_3, predictors_4,
                   predictors_5, predictors_6, predictors_7, predictors_8]

cat_features = [None, None, None, ['sleep', 'wake'], ['sleep_hours', 'sleep_wakes'],
                ['lang_36'], ['sleep', 'wake', 'sleep_hours', 'sleep_wakes', 'lang_36'],
                ['lang_36'], ['sleep', 'wake', 'sleep_hours', 'sleep_wakes', 'lang_36']]

# if wake and sleep variables will be used as categorical (possible), do not standardize
STANDARDIZE_VARS = ['age_father', 'age_mother', 'wake', 'sleep', 'asq_score', 'icq_score']


def standardize(df_train, df_test, var_stand):
    for var in var_stand:
        if (var not in df_train.columns) or (var not in df_test.columns):
            continue
        mean_var = df_train[var].mean()
        sd_var = df_train[var].std()
        if pd.isna(sd_var) or (sd_var == 0):
            continue
        df_train[var] = (df_train[var] - mean_var) / sd_var
        df_test[var] = (df_test[var] - mean_var) / sd_var


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def load_dataframe(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(path)

    raise ValueError(f'Unsupported file format: {suffix}')

# Assumes that the target is suitable for stratification; may fail in highly imbalanced datasets
def get_splitter(train_df, target, group_col, n_splits, random_state):
    if group_col and (group_col in train_df.columns):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Keeping only predictors present in both train and test; be aware of potential silent shrikage of predictor sets
def get_split_iter(splitter, train_df, target, predictors, group_col):
    X = train_df[predictors]
    y = train_df[target]

    if group_col and (group_col in train_df.columns):
        groups = train_df[group_col]
        return splitter.split(X, y, groups)

    return splitter.split(X, y)


def get_available_predictors(df_a, df_b, predictors):
    return [p for p in predictors if (p in df_a.columns) and (p in df_b.columns)]


def get_available_cat_features(cat_vars, predictors):
    if cat_vars is None:
        return None
    available = [c for c in cat_vars if c in predictors]
    return available if len(available) > 0 else None


def tune_catboost(train_df, valid_df, predictors, cat_vars, target='F84', group_col='famID',
                  random_state=42, n_splits=5):
    iterations_grid = np.array([1500])
    depth_grid = np.array([3, 4, 5, 6])
    learning_rate_grid = np.array([0.01, 0.03, 0.05])
    l2_leaf_reg_grid = np.array([3, 10, 15])

    metaparams = cartesian_product(iterations_grid, depth_grid, learning_rate_grid, l2_leaf_reg_grid)

    splitter = get_splitter(train_df, target, group_col, n_splits, random_state)

    tuning_progress = []
    best_params = {}
    best_score = 0

    for i in range(0, metaparams.shape[0]):
        iterations = int(metaparams[i][0])
        depth = int(metaparams[i][1])
        learning_rate = float(metaparams[i][2])
        l2_leaf_reg = float(metaparams[i][3])

        print(f'{i + 1} out of {metaparams.shape[0]}:', iterations, depth, learning_rate, l2_leaf_reg)

        auc_cv = []
        tree_count = []

        split_iter = get_split_iter(splitter, train_df, target, predictors, group_col)

        for fold_train_index, fold_valid_index in split_iter:
            train_fold = train_df.iloc[fold_train_index].copy()
            valid_fold = train_df.iloc[fold_valid_index].copy()

            clf = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                verbose=False,
                random_state=random_state,
                custom_metric=[metrics.AUC()]
            )

            clf.fit(
                train_fold[predictors],
                train_fold[target],
                cat_features=cat_vars,
                eval_set=(valid_fold[predictors], valid_fold[target]),
                use_best_model=True,
                early_stopping_rounds=50
            )

            auc_model = clf.get_best_score().get('validation').get('AUC')
            auc_cv.append(auc_model)
            tree_count.append(clf.tree_count_)

        auc_mean = np.mean(auc_cv)
        auc_sd = np.std(auc_cv)

        tuning_progress.append({
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'auc_mean': auc_mean,
            'auc_sd': auc_sd,
            'max_tree_count': max(tree_count),
            'min_tree_count': min(tree_count)
        })

        if (auc_mean > best_score) and (max(tree_count) > 100):
            best_score = auc_mean
            best_params = {
                'iterations': iterations,
                'depth': depth,
                'learning_rate': learning_rate,
                'l2_leaf_reg': l2_leaf_reg
            }

    clf = CatBoostClassifier(
        iterations=int(best_params['iterations']),
        depth=int(best_params['depth']),
        learning_rate=float(best_params['learning_rate']),
        l2_leaf_reg=float(best_params['l2_leaf_reg']),
        verbose=False,
        random_state=random_state,
        custom_metric=[metrics.AUC()]
    )

    clf.fit(
        train_df[predictors],
        train_df[target],
        cat_features=cat_vars
    )

    y_train = train_df[target].copy()
    y_valid = valid_df[target].copy()

    y_pred_proba_train = clf.predict_proba(train_df[predictors])[:, 1]
    y_pred_proba_valid = clf.predict_proba(valid_df[predictors])[:, 1]
    y_pred_valid = (y_pred_proba_valid > 0.5).astype(int)

    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_valid = roc_auc_score(y_valid, y_pred_proba_valid)
    bal_acc_valid = balanced_accuracy_score(y_valid, y_pred_valid)

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba_valid)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    thresholdOpt = thresholds[ix]
    y_pred_valid_opt = (y_pred_proba_valid > thresholdOpt).astype(int)
    bal_acc_valid_opt = balanced_accuracy_score(y_valid, y_pred_valid_opt)

    pred_df = valid_df[[target]].copy()
    pred_df['pred_proba'] = y_pred_proba_valid
    pred_df['pred_class_0_5'] = y_pred_valid
    pred_df['pred_class_opt'] = y_pred_valid_opt

    feature_importance = clf.get_feature_importance(prettified=True)

    metrics_dict = {
        'auc_train': auc_train,
        'auc_valid': auc_valid,
        'balanced_accuracy_valid_0_5': bal_acc_valid,
        'balanced_accuracy_valid_opt_gmean': bal_acc_valid_opt,
        'threshold_opt_valid': float(thresholdOpt)
    }

    return clf, best_params, pd.DataFrame(tuning_progress), metrics_dict, pred_df, feature_importance


def fit_final_model(trainValid, test, predictors, cat_vars, best_params, thresholdOpt,
                    target='F84', random_state=42):
    clf = CatBoostClassifier(
        iterations=int(best_params['iterations']),
        depth=int(best_params['depth']),
        learning_rate=float(best_params['learning_rate']),
        l2_leaf_reg=float(best_params['l2_leaf_reg']),
        verbose=False,
        random_state=random_state,
        custom_metric=[metrics.AUC()]
    )

    clf.fit(
        trainValid[predictors],
        trainValid[target],
        cat_features=cat_vars
    )

    y_trainValid = trainValid[target].copy()
    y_test = test[target].copy()

    y_pred_proba_trainValid = clf.predict_proba(trainValid[predictors])[:, 1]
    y_pred_proba_test = clf.predict_proba(test[predictors])[:, 1]

    y_pred_test_0_5 = (y_pred_proba_test > 0.5).astype(int)
    y_pred_test_opt = (y_pred_proba_test > thresholdOpt).astype(int)

    auc_trainValid = roc_auc_score(y_trainValid, y_pred_proba_trainValid)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    bal_acc_test = balanced_accuracy_score(y_test, y_pred_test_0_5)
    bal_acc_test_opt = balanced_accuracy_score(y_test, y_pred_test_opt)

    pred_df = test[[target]].copy()
    pred_df['pred_proba'] = y_pred_proba_test
    pred_df['pred_class_0_5'] = y_pred_test_0_5
    pred_df['pred_class_opt'] = y_pred_test_opt

    feature_importance = clf.get_feature_importance(prettified=True)

    metrics_dict = {
        'auc_trainValid': auc_trainValid,
        'auc_test': auc_test,
        'balanced_accuracy_test_0_5': bal_acc_test,
        'balanced_accuracy_test_opt_gmean': bal_acc_test_opt,
        'threshold_opt_valid': float(thresholdOpt)
    }

    return clf, metrics_dict, pred_df, feature_importance


def save_feature_importance_plot(feature_importance, outfile, top_n=20):
    data = feature_importance.head(top_n).copy()

    plt.figure(figsize=(8, max(6, int(top_n * 0.35))))
    sns.barplot(y='Feature Id', x='Importances', hue='Feature Id', legend=False,
                data=data, palette='flare')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Clean CatBoost pipeline with train, valid, and test.')
    parser.add_argument('--train', required=True, help='Path to training data file')
    parser.add_argument('--valid', required=True, help='Path to validation data file')
    parser.add_argument('--test', required=True, help='Path to test data file')
    parser.add_argument('--outdir', required=True, help='Directory to save outputs')
    parser.add_argument('--target', default='F84', help='Target column')
    parser.add_argument('--group-col', default='famID', help='Grouping column for StratifiedGroupKFold')
    parser.add_argument('--predictor-set', default='all',
                        help='Predictor set index (0-8) or "all"')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = load_dataframe(args.train)
    valid = load_dataframe(args.valid)
    test = load_dataframe(args.test)

    standardize(train, valid, STANDARDIZE_VARS)
    standardize(train, test, STANDARDIZE_VARS)

    if args.predictor_set == 'all':
        predictor_ids = list(range(len(predictors_list)))
    else:
        predictor_ids = [int(args.predictor_set)]

    model_summary = []

    for predictor_id in predictor_ids:
        predictors = predictors_list[predictor_id]
        cat_vars = cat_features[predictor_id]

        predictors = get_available_predictors(train, valid, predictors)
        predictors = get_available_predictors(pd.DataFrame(columns=predictors), test, predictors)
        cat_vars = get_available_cat_features(cat_vars, predictors)

        if len(predictors) == 0:
            continue

        clf_valid, best_params, tuning_df, valid_metrics, valid_pred_df, feature_importance = tune_catboost(
            train_df=train,
            valid_df=valid,
            predictors=predictors,
            cat_vars=cat_vars,
            target=args.target,
            group_col=args.group_col,
            random_state=args.random_state,
            n_splits=args.n_splits
        )

        tuning_df.to_csv(outdir / f'tuning_predictors_{predictor_id}.csv', index=False)
        valid_pred_df.to_csv(outdir / f'valid_predictions_predictors_{predictor_id}.csv', index=False)
        feature_importance.to_csv(outdir / f'feature_importance_predictors_{predictor_id}.csv', index=False)
        save_feature_importance_plot(
            feature_importance,
            outdir / f'feature_importance_predictors_{predictor_id}.png'
        )

        summary_row = {
            'predictor_id': predictor_id,
            'n_predictors': len(predictors),
            'predictors': predictors,
            'cat_vars': cat_vars,
            **best_params,
            **valid_metrics
        }
        model_summary.append(summary_row)

        with open(outdir / f'model_summary_predictors_{predictor_id}.json', 'w') as f:
            json.dump(summary_row, f, indent=2)

    model_summary_df = pd.DataFrame(model_summary).sort_values('auc_valid', ascending=False)
    model_summary_df.to_csv(outdir / 'model_summary.csv', index=False)

    best_overall = model_summary_df.iloc[0].to_dict()
    best_predictor_id = int(best_overall['predictor_id'])

    best_predictors = predictors_list[best_predictor_id]
    best_predictors = get_available_predictors(train, valid, best_predictors)
    best_predictors = get_available_predictors(pd.DataFrame(columns=best_predictors), test, best_predictors)
    best_cat_vars = get_available_cat_features(cat_features[best_predictor_id], best_predictors)

    best_params = {
        'iterations': int(best_overall['iterations']),
        'depth': int(best_overall['depth']),
        'learning_rate': float(best_overall['learning_rate']),
        'l2_leaf_reg': float(best_overall['l2_leaf_reg'])
    }
    thresholdOpt = float(best_overall['threshold_opt_valid'])

    trainValid = pd.concat([train, valid], axis=0).reset_index(drop=True)

    clf_test, test_metrics, test_pred_df, test_feature_importance = fit_final_model(
        trainValid=trainValid,
        test=test,
        predictors=best_predictors,
        cat_vars=best_cat_vars,
        best_params=best_params,
        thresholdOpt=thresholdOpt,
        target=args.target,
        random_state=args.random_state
    )

    test_pred_df.to_csv(outdir / f'test_predictions_predictors_{best_predictor_id}.csv', index=False)
    test_feature_importance.to_csv(outdir / f'feature_importance_test_predictors_{best_predictor_id}.csv',
                                   index=False)
    save_feature_importance_plot(
        test_feature_importance,
        outdir / f'feature_importance_test_predictors_{best_predictor_id}.png'
    )

    best_model_summary = {
        **best_overall,
        **test_metrics
    }

    with open(outdir / 'best_model_summary.json', 'w') as f:
        json.dump(best_model_summary, f, indent=2)


if __name__ == '__main__':
    main()
