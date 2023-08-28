import platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from trading.configuration import TimeRange
from trading.data.dataprovider import DataProvider
from trading.tradingai.data_drawer import FreqaiDataDrawer
from trading.tradingai.data_kitchen import FreqaiDataKitchen
from trading.resolvers import StrategyResolver
from trading.resolvers.tradingaimodel_resolver import FreqaiModelResolver
from tests.conftest import get_patched_exchange


def is_mac() -> bool:
    machine = platform.system()
    return "Darwin" in machine


@pytest.fixture(scope="function")
def tradingai_conf(default_conf, tmpdir):
    tradingaiconf = deepcopy(default_conf)
    tradingaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "strategy": "tradingai_test_strat",
            "user_data_dir": Path(tmpdir),
            "strategy-path": "trading/tests/strategy/strats",
            "tradingaimodel": "LightGBMRegressor",
            "tradingaimodel_path": "tradingai/prediction_models",
            "timerange": "20180110-20180115",
            "tradingai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "uniqe-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path('config_examples', 'config_tradingai.example.json')]
        }
    )
    tradingaiconf['exchange'].update({'pair_whitelist': ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC']})
    return tradingaiconf


def make_rl_config(conf):
    conf.update({"strategy": "tradingai_rl_test_strat"})
    conf["tradingai"].update({"model_training_parameters": {
        "learning_rate": 0.00025,
        "gamma": 0.9,
        "verbose": 1
    }})
    conf["tradingai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {
            "rr": 1,
            "profit_aim": 0.02,
            "win_reward_factor": 2
        },
        "drop_ohlc_from_features": False
        }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> Dict[str, Any]:
    return {
            "learning_rate": 3e-4,
            "trainer_kwargs": {
                "n_steps": None,
                "batch_size": 64,
                "n_epochs": 1,
            },
            "model_kwargs": {
                "hidden_dim": 32,
                "dropout_percent": 0.2,
                "n_layer": 1,
            }
        }


def get_patched_data_kitchen(mocker, tradingaiconf):
    dk = FreqaiDataKitchen(tradingaiconf)
    return dk


def get_patched_data_drawer(mocker, tradingaiconf):
    # dd = mocker.patch('trading.tradingai.data_drawer', MagicMock())
    dd = FreqaiDataDrawer(tradingaiconf)
    return dd


def get_patched_tradingai_strategy(mocker, tradingaiconf):
    strategy = StrategyResolver.load_strategy(tradingaiconf)
    strategy.ft_bot_start()

    return strategy


def get_patched_tradingaimodel(mocker, tradingaiconf):
    tradingaimodel = FreqaiModelResolver.load_tradingaimodel(tradingaiconf)

    return tradingaimodel


def make_unfiltered_dataframe(mocker, tradingai_conf):
    tradingai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    tradingai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(data_load_timerange, tradingai.dk)

    tradingai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = tradingai.dd.get_base_and_corr_dataframes(
            data_load_timerange, tradingai.dk.pair, tradingai.dk
        )

    unfiltered_dataframe = tradingai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, tradingai.dk.pair
            )
    for i in range(5):
        unfiltered_dataframe[f'constant_{i}'] = i

    unfiltered_dataframe = tradingai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return tradingai, unfiltered_dataframe


def make_data_dictionary(mocker, tradingai_conf):
    tradingai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    tradingai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    tradingai.dd.load_all_pair_histories(data_load_timerange, tradingai.dk)

    tradingai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = tradingai.dd.get_base_and_corr_dataframes(
            data_load_timerange, tradingai.dk.pair, tradingai.dk
        )

    unfiltered_dataframe = tradingai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, tradingai.dk.pair
            )

    unfiltered_dataframe = tradingai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    tradingai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = tradingai.dk.filter_features(
            unfiltered_dataframe,
            tradingai.dk.training_features_list,
            tradingai.dk.label_list,
            training_filter=True,
        )

    data_dictionary = tradingai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = tradingai.dk.normalize_data(data_dictionary)

    return tradingai


def get_tradingai_live_analyzed_dataframe(mocker, tradingaiconf):
    strategy = get_patched_tradingai_strategy(mocker, tradingaiconf)
    exchange = get_patched_exchange(mocker, tradingaiconf)
    strategy.dp = DataProvider(tradingaiconf, exchange)
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingaiconf, tradingai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair('ADA/BTC', '5m')
    return strategy.dp.get_analyzed_dataframe('ADA/BTC', '5m')


def get_tradingai_analyzed_dataframe(mocker, tradingaiconf):
    strategy = get_patched_tradingai_strategy(mocker, tradingaiconf)
    exchange = get_patched_exchange(mocker, tradingaiconf)
    strategy.dp = DataProvider(tradingaiconf, exchange)
    strategy.tradingai_info = tradingaiconf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingaiconf, tradingai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradingai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return tradingai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')


def get_ready_to_train(mocker, tradingaiconf):
    strategy = get_patched_tradingai_strategy(mocker, tradingaiconf)
    exchange = get_patched_exchange(mocker, tradingaiconf)
    strategy.dp = DataProvider(tradingaiconf, exchange)
    strategy.tradingai_info = tradingaiconf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingaiconf, tradingai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradingai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, tradingai, strategy
