from dataclasses import asdict, dataclass
import pickle
import json
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class AdvancedXGBoostConfig:
    # Core parameters
    n_estimators: int = 10000
    learning_rate: float = 0.05
    max_depth: int = 8
    min_child_weight: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 0.8
    colsample_bynode: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    gamma: float = 0.1
    grow_policy: str = 'depthwise'
    tree_method: str = 'hist'
    booster: str = 'gbtree'
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    early_stopping_rounds: int = 100
    # Advanced parameters
    monotone_constraints: tuple = None
    interaction_constraints: list = None
    num_parallel_tree: int = 1
    max_bin: int = 512
    scale_pos_weight: float = 1.0
    base_score: float = None
    missing: float = np.nan
    enable_categorical: bool = False
    callbacks: list = None
    # Custom extensions
    use_gpu: bool = False
    shap_calculation: bool = True
    hyperopt_iter: int = 0
    cv_folds: int = 5

class AdvancedXGBoost:
    def __init__(self, config: AdvancedXGBoostConfig):
        self.config = asdict(config)
        self._prepare_model()
        self.shap_explainer = None
        self.feature_importances_ = None
        self.best_iteration = 0
        self.cv_results = None
        
    def _prepare_model(self):
        """Configure GPU support and advanced parameters"""
        if self.config['use_gpu'] and xgb.__version__ >= '1.5.0':
            self.config.update({'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
            
        self.model = xgb.XGBClassifier(
            **{k:v for k,v in self.config.items() 
               if k in xgb.XGBClassifier().get_params().keys()},
            use_label_encoder=False,
            verbosity=0
        )
        
        self.custom_params = {
            'monotone_constraints': self.config['monotone_constraints'],
            'interaction_constraints': self.config['interaction_constraints'],
            'num_parallel_tree': self.config['num_parallel_tree'],
            'callbacks': [xgb.callback.EarlyStopping(
                rounds=self.config['early_stopping_rounds']
            )]
        }

    def _hyperparameter_tuning(self, X, y):
        """Bayesian optimization with Hyperopt"""
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'max_depth': hp.quniform('max_depth', 3, 12, 1),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100)),
            'gamma': hp.loguniform('gamma', np.log(1e-5), np.log(10))
        }

        def objective(params):
            cv_results = xgb.cv(
                params,
                xgb.DMatrix(X, y),
                num_boost_round=self.config['n_estimators'],
                nfold=5,
                metrics={self.config['eval_metric']},
                early_stopping_rounds=self.config['early_stopping_rounds'],
                stratified=True,
                seed=42
            )
            return cv_results['test-{}-mean'.format(self.config['eval_metric'])].iloc[-1]

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.config['hyperopt_iter'],
                    trials=trials)
        
        self.config.update(best)
        self._prepare_model()

    def fit(self, X, y, eval_set=None, sample_weight=None, class_weight=None, verbose=100):
        """Enhanced fitting with multiple advanced options"""
        if self.config['hyperopt_iter'] > 0:
            self._hyperparameter_tuning(X, y)
            
        if class_weight is not None:
            sample_weight = self._calculate_weights(y, class_weight)
            
        if self.config['cv_folds'] > 1:
            self._cross_validate(X, y, sample_weight)
        else:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                sample_weight=sample_weight,
                verbose=verbose,
                **self.custom_params
            )
            
        self._post_fit_processing(X, y)
        
    def _cross_validate(self, X, y, sample_weight):
        """Stratified k-fold cross-validation with pruning"""
        skf = StratifiedKFold(n_splits=self.config['cv_folds'])
        cv_results = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**self.model.get_params())
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
                **self.custom_params
            )
            cv_results.append(model.evals_result())
            
        self.cv_results = cv_results

    def _post_fit_processing(self, X, y):
        """SHAP values, feature importance, and model analysis"""
        if self.config['shap_calculation']:
            self.shap_explainer = shap.Explainer(self.model)
            self.shap_values = self.shap_explainer(X)
            
        self.feature_importances_ = self.model.feature_importances_
        self.best_iteration = self.model.best_iteration
        
    def predict(self, X, output_margin=False, pred_contribs=False):
        """Enhanced prediction with multiple output types"""
        if pred_contribs and self.shap_explainer:
            return self.shap_explainer(X)
        return self.model.predict(X, output_margin=output_margin)

    def predict_proba(self, X, iteration_range=(0, 0)):
        return self.model.predict_proba(X, iteration_range=iteration_range)

    def save(self, path, format='pkl'):
        """Multi-format saving with version control"""
        if format == 'pkl':
            pickle.dump(self, open(path, 'wb'))
        elif format == 'json':
            self.model.save_model(path)
            with open(path + '.config', 'w') as f:
                json.dump(self.config, f)
        elif format == 'ubj':
            self.model.save_model(path, format='ubj')

    @classmethod
    def load(cls, path, format='pkl'):
        if format == 'pkl':
            return pickle.load(open(path, 'rb'))
        elif format == 'json':
            model = cls.__new__(cls)
            model.model = xgb.XGBClassifier()
            model.model.load_model(path)
            with open(path + '.config', 'r') as f:
                model.config = json.load(f)
            return model

    def get_feature_importance(self, importance_type='weight'):
        return self.model.get_booster().get_score(importance_type=importance_type)

    def partial_dependence_plot(self, X, features):
        return shap.partial_dependence_plot(
            features, self.model.predict, X, model_expected_value=True
        )

    def _calculate_weights(self, y, class_weight):
        return np.array([class_weight[label] for label in y])

    def plot_learning_curve(self):
        return xgb.plot_metric(self.model.evals_result())

    def statistical_analysis(self, X, y):
        """Compare feature distributions between predicted classes"""
        preds = self.predict(X)
        results = {}
        for feature in X.columns:
            stat, p = stats.ttest_ind(
                X[feature][preds == 0],
                X[feature][preds == 1],
                equal_var=False
            )
            results[feature] = {'t-statistic': stat, 'p-value': p}
        return results

    def adversarial_validation(self, X_train, X_test):
        """Detect data drift between train and test sets"""
        combined = np.concatenate([X_train, X_test])
        labels = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
        val_model = AdvancedXGBoost(AdvancedXGBoostConfig(objective='binary:logistic'))
        val_model.fit(combined, labels)
        return val_model.model.predict_proba(combined)[:, 1]