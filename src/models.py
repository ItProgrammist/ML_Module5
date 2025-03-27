from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from catboost import CatBoostClassifier, Pool
import optuna
import logging

def train_random_forest(X_train, X_test, y_train, y_test):

    # Random Forest
    clf_rf = RandomForestClassifier()

    parameters = {
        'n_estimators': [10, 20, 30],
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [10, 20, 30, 40, 50],
        'min_samples_split': [3, 5, 7, 10]
    }
    grid_search_cv_rf_clf = GridSearchCV(clf_rf, parameters, cv=5)
    grid_search_cv_rf_clf.fit(X_train, y_train)
    best_rf_clf = grid_search_cv_rf_clf.best_estimator_
    y_pred2 = best_rf_clf.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred2)

    return best_rf_clf, accuracy1


def objective(trial, X_train, X_test, y_train, y_test):
    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    parameters_cb = {
        'iterations': trial.suggest_categorical('iterations', [90, 100]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
        'depth': trial.suggest_categorical('depth', [4, 6]),
        'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1e-3, 1e-2, 0.1]),
        'bagging_temperature': trial.suggest_categorical('bagging_temperature', [0.1, 0.3, 0.5]),
        'random_strength': trial.suggest_categorical('random_strength', [0.5, 1.0, 2.0]),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0
    }

    catboost_clf = CatBoostClassifier(**parameters_cb)

    catboost_clf.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=0)

    y_pred = catboost_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def train_catboost(X_train, X_test, y_train, y_test):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=50)

    logging.info(f"Best trial: {study.best_trial.params}")
    logging.info(f"Best accuracy: {study.best_value}")

    best_params = study.best_trial.params
    best_catboost_clf = CatBoostClassifier(**best_params)
    best_catboost_clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)

    y_pred = best_catboost_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return best_catboost_clf, accuracy
