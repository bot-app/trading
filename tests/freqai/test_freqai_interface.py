import logging
import platform
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading.configuration import TimeRange
from trading.data.dataprovider import DataProvider
from trading.enums import RunMode
from trading.tradingai.data_kitchen import FreqaiDataKitchen
from trading.tradingai.utils import download_all_data_for_training, get_required_data_timerange
from trading.optimize.backtesting import Backtesting
from trading.persistence import Trade
from trading.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, log_has_re
from tests.tradingai.conftest import (get_patched_tradingai_strategy, is_mac, make_rl_config,
                                   mock_pytorch_mlp_model_training_parameters)


def is_py11() -> bool:
    return sys.version_info >= (3, 11)


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


def can_run_model(model: str) -> None:
    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model
    if is_pytorch_model and is_mac() and not is_arm():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    ('LightGBMRegressor', True, False, True, True, False, 0, 0),
    ('XGBoostRegressor', False, True, False, True, False, 10, 0.05),
    ('XGBoostRFRegressor', False, False, False, True, False, 0, 0),
    ('CatboostRegressor', False, False, False, True, True, 0, 0),
    ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0),
    ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner', False, True, False, True, False, 0, 0),
    ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0),
    ])
def test_extract_data_and_train_model_Standard(mocker, tradingai_conf, model, pca,
                                               dbscan, float32, can_short, shuffle,
                                               buffer, noise):

    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = 'joblib'
    tradingai_conf.update({"tradingaimodel": model})
    tradingai_conf.update({"timerange": "20180110-20180130"})
    tradingai_conf.update({"strategy": "tradingai_test_strat"})
    tradingai_conf['tradingai']['feature_parameters'].update({"principal_component_analysis": pca})
    tradingai_conf['tradingai']['feature_parameters'].update({"use_DBSCAN_to_remove_outliers": dbscan})
    tradingai_conf.update({"reduce_df_footprint": float32})
    tradingai_conf['tradingai']['feature_parameters'].update({"shuffle_after_split": shuffle})
    tradingai_conf['tradingai']['feature_parameters'].update({"buffer_train_data_candles": buffer})
    tradingai_conf['tradingai']['feature_parameters'].update({"noise_standard_deviation": noise})

    if 'ReinforcementLearner' in model:
        model_save_ext = 'zip'
        tradingai_conf = make_rl_config(tradingai_conf)
        # test the RL guardrails
        tradingai_conf['tradingai']['feature_parameters'].update({"use_SVM_to_remove_outliers": True})
        tradingai_conf['tradingai']['feature_parameters'].update({"DI_threshold": 2})
        tradingai_conf['tradingai']['data_split_parameters'].update({'shuffle': True})

    if 'test_3ac' in model or 'test_4ac' in model:
        tradingai_conf["tradingaimodel_path"] = str(Path(__file__).parents[1] / "tradingai" / "test_models")
        tradingai_conf["tradingai"]["rl_config"]["drop_ohlc_from_features"] = True

    if 'PyTorch' in model:
        model_save_ext = 'zip'
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradingai_conf['tradingai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            tradingai_conf.update({"conv_width": 10})

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.activate_tensorboard = test_tb
    tradingai.can_short = can_short
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    tradingai.dk.set_paths('ADA/BTC', 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)

    tradingai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    tradingai.dk.set_paths('ADA/BTC', None)

    tradingai.train_timer("start", "ADA/BTC")
    tradingai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradingai.dk, data_load_timerange)
    tradingai.train_timer("stop", "ADA/BTC")
    tradingai.dd.save_metric_tracker_to_disk()
    tradingai.dd.save_drawer_to_disk()

    assert Path(tradingai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(tradingai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(tradingai.dk.data_path /
                f"{tradingai.dk.model_filename}_model.{model_save_ext}").is_file()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_metadata.json").is_file()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(tradingai.dk.full_path))


@pytest.mark.parametrize('model, strat', [
    ('LightGBMRegressorMultiTarget', "tradingai_test_multimodel_strat"),
    ('XGBoostRegressorMultiTarget', "tradingai_test_multimodel_strat"),
    ('CatboostRegressorMultiTarget', "tradingai_test_multimodel_strat"),
    ('LightGBMClassifierMultiTarget', "tradingai_test_multimodel_classifier_strat"),
    ('CatboostClassifierMultiTarget', "tradingai_test_multimodel_classifier_strat")
    ])
def test_extract_data_and_train_model_MultiTargets(mocker, tradingai_conf, model, strat):
    can_run_model(model)

    tradingai_conf.update({"timerange": "20180110-20180130"})
    tradingai_conf.update({"strategy": strat})
    tradingai_conf.update({"tradingaimodel": model})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)

    tradingai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradingai.dk.set_paths('ADA/BTC', None)

    tradingai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradingai.dk, data_load_timerange)

    assert len(tradingai.dk.label_list) == 2
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_model.joblib").is_file()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_metadata.json").is_file()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(tradingai.dk.data['training_features_list']) == 14

    shutil.rmtree(Path(tradingai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    'XGBoostRFClassifier',
    'PyTorchMLPClassifier',
    ])
def test_extract_data_and_train_model_Classifiers(mocker, tradingai_conf, model):
    can_run_model(model)

    tradingai_conf.update({"tradingaimodel": model})
    tradingai_conf.update({"strategy": "tradingai_test_classifier"})
    tradingai_conf.update({"timerange": "20180110-20180130"})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)

    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)

    tradingai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradingai.dk.set_paths('ADA/BTC', None)

    tradingai.extract_data_and_train_model(new_timerange, "ADA/BTC",
                                        strategy, tradingai.dk, data_load_timerange)

    if 'PyTorchMLPClassifier':
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradingai_conf['tradingai']['model_training_parameters'].update(pytorch_mlp_mtp)

    if tradingai.dd.model_type == 'joblib':
        model_file_extension = ".joblib"
    elif tradingai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(f"Unsupported model type: {tradingai.dd.model_type},"
                        f" can't assign model_file_extension")

    assert Path(tradingai.dk.data_path /
                f"{tradingai.dk.model_filename}_model{model_file_extension}").exists()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_metadata.json").exists()
    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(tradingai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "tradingai_test_strat"),
        ("XGBoostRegressor", 2, "tradingai_test_strat"),
        ("CatboostRegressor", 2, "tradingai_test_strat"),
        ("PyTorchMLPRegressor", 2, "tradingai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "tradingai_test_strat"),
        ("ReinforcementLearner", 3, "tradingai_rl_test_strat"),
        ("XGBoostClassifier", 2, "tradingai_test_classifier"),
        ("LightGBMClassifier", 2, "tradingai_test_classifier"),
        ("CatboostClassifier", 2, "tradingai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "tradingai_test_classifier")
    ],
    )
def test_start_backtesting(mocker, tradingai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac():
        test_tb = False

    tradingai_conf.get("tradingai", {}).update({"save_backtest_models": True})
    tradingai_conf['runmode'] = RunMode.BACKTEST

    Trade.use_db = False

    tradingai_conf.update({"tradingaimodel": model})
    tradingai_conf.update({"timerange": "20180120-20180130"})
    tradingai_conf.update({"strategy": strat})

    if 'ReinforcementLearner' in model:
        tradingai_conf = make_rl_config(tradingai_conf)

    if 'test_4ac' in model:
        tradingai_conf["tradingaimodel_path"] = str(Path(__file__).parents[1] / "tradingai" / "test_models")

    if 'PyTorch' in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        tradingai_conf['tradingai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            tradingai_conf.update({"conv_width": 10})

    tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.activate_tensorboard = test_tb
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)
    df = base_df[tradingai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    tradingai.dk.set_paths('LTC/BTC', None)
    tradingai.start_backtesting(df, metadata, tradingai.dk, strategy)
    model_folders = [x for x in tradingai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, tradingai_conf):
    tradingai_conf.update({"timerange": "20180120-20180124"})
    tradingai_conf.get("tradingai", {}).update({"backtest_period_days": 0.5})
    tradingai_conf.get("tradingai", {}).update({"save_backtest_models": True})
    tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)
    df = base_df[tradingai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    tradingai.start_backtesting(df, metadata, tradingai.dk, strategy)
    model_folders = [x for x in tradingai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(tradingai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, tradingai_conf, caplog):
    tradingai_conf.update({"timerange": "20180120-20180130"})
    tradingai_conf.get("tradingai", {}).update({"save_backtest_models": True})
    tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)
    df = base_df[tradingai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    tradingai.dk.pair = pair
    tradingai.start_backtesting(df, metadata, tradingai.dk, strategy)
    model_folders = [x for x in tradingai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    tradingai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)
    df = base_df[tradingai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    tradingai.dk.pair = pair
    tradingai.start_backtesting(df, metadata, tradingai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    tradingai.dk.pair = pair
    tradingai.start_backtesting(df, metadata, tradingai.dk, strategy)

    path = (tradingai.dd.full_path / tradingai.dk.backtest_predictions_folder)
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(tradingai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, tradingai_conf, caplog):
    tradingai_conf.get("tradingai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)
    df = tradingai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_tradingai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = tradingai.dk.remove_special_chars_from_feature_names(df)
    tradingai.dk.get_unique_classes_from_labels(df)
    tradingai.dk.pair = "ADA/BTC"
    tradingai.dk.full_df = df.fillna(0)
    tradingai.dk.full_df
    assert "&-s_close_mean" not in tradingai.dk.full_df.columns
    assert "&-s_close_std" not in tradingai.dk.full_df.columns
    tradingai.backtesting_fit_live_predictions(tradingai.dk)
    assert "&-s_close_mean" in tradingai.dk.full_df.columns
    assert "&-s_close_std" in tradingai.dk.full_df.columns
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_plot_feature_importance(mocker, tradingai_conf):

    from trading.tradingai.utils import plot_feature_importance

    tradingai_conf.update({"timerange": "20180110-20180130"})
    tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"})

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)

    tradingai.dd.pair_dict = {"ADA/BTC": {"model_filename": "fake_name",
                                       "trained_timestamp": 1, "data_path": "", "extras": {}}}

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    tradingai.dk.set_paths('ADA/BTC', None)

    tradingai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, tradingai.dk, data_load_timerange)

    model = tradingai.dd.load_data("ADA/BTC", tradingai.dk)

    plot_feature_importance(model, "ADA/BTC", tradingai.dk)

    assert Path(tradingai.dk.data_path / f"{tradingai.dk.model_filename}.html")

    shutil.rmtree(Path(tradingai.dk.full_path))


@pytest.mark.parametrize('timeframes,corr_pairs', [
    (['5m'], ['ADA/BTC', 'DASH/BTC']),
    (['5m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
    (['5m', '15m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
])
def test_tradingai_informative_pairs(mocker, tradingai_conf, timeframes, corr_pairs):
    tradingai_conf['tradingai']['feature_parameters'].update({
        'include_timeframes': timeframes,
        'include_corr_pairlist': corr_pairs,

    })
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    pairlists = PairListManager(exchange, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, tradingai_conf, caplog):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    pairlist = PairListManager(exchange, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange, pairlist)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = False

    tradingai.train_queue = tradingai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, tradingai_conf):
    time_range = get_required_data_timerange(tradingai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, tradingai_conf, caplog, tmpdir):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    pairlist = PairListManager(exchange, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange, pairlist)
    tradingai_conf['pairs'] = tradingai_conf['exchange']['pair_whitelist']
    tradingai_conf['datadir'] = Path(tmpdir)
    download_all_data_for_training(strategy.dp, tradingai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('dp_exists', [(False), (True)])
def test_get_state_info(mocker, tradingai_conf, dp_exists, caplog, tickers):

    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")
    if is_py11():
        pytest.skip("Reinforcement learning currently not available on python 3.11.")

    tradingai_conf.update({"tradingaimodel": "ReinforcementLearner"})
    tradingai_conf.update({"timerange": "20180110-20180130"})
    tradingai_conf.update({"strategy": "tradingai_rl_test_strat"})
    tradingai_conf = make_rl_config(tradingai_conf)
    tradingai_conf['entry_pricing']['price_side'] = 'same'
    tradingai_conf['exit_pricing']['price_side'] = 'same'

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    ticker_mock = MagicMock(return_value=tickers()['ETH/BTC'])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(tradingai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.data_provider = strategy.dp
    tradingai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    tradingai.get_state_info("ADA/BTC")
    tradingai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )
