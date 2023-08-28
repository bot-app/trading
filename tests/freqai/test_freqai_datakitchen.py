import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading.configuration import TimeRange
from trading.data.dataprovider import DataProvider
from trading.exceptions import OperationalException
from trading.tradingai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.tradingai.conftest import (get_patched_data_kitchen, get_patched_tradingai_strategy,
                                   make_unfiltered_dataframe)
from tests.tradingai.test_tradingai_interface import is_mac


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, tradingai_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, tradingai_conf)
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result
    shutil.rmtree(Path(dk.full_path))


def test_create_fulltimerange_incorrect_backtest_period(mocker, tradingai_conf):
    dk = get_patched_data_kitchen(mocker, tradingai_conf)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)
    shutil.rmtree(Path(dk.full_path))


@pytest.mark.parametrize(
    "timerange, train_period_days, backtest_period_days, expected_result",
    [
        ("20220101-20220201", 30, 7, 9),
        ("20220101-20220201", 30, 0.5, 120),
        ("20220101-20220201", 10, 1, 80),
    ],
)
def test_split_timerange(
    mocker, tradingai_conf, timerange, train_period_days, backtest_period_days, expected_result
):
    tradingai_conf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, tradingai_conf)
    tr_list, bt_list = dk.split_timerange(timerange, train_period_days, backtest_period_days)
    assert len(tr_list) == len(bt_list) == expected_result

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)
    shutil.rmtree(Path(dk.full_path))


def test_check_if_model_expired(mocker, tradingai_conf):

    dk = get_patched_data_kitchen(mocker, tradingai_conf)
    now = datetime.now(tz=timezone.utc).timestamp()
    assert dk.check_if_model_expired(now) is False
    now = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).timestamp()
    assert dk.check_if_model_expired(now) is True
    shutil.rmtree(Path(dk.full_path))


def test_filter_features(mocker, tradingai_conf):
    tradingai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, tradingai_conf)
    tradingai.dk.find_features(unfiltered_dataframe)

    filtered_df, labels = tradingai.dk.filter_features(
            unfiltered_dataframe,
            tradingai.dk.training_features_list,
            tradingai.dk.label_list,
            training_filter=True,
    )

    assert len(filtered_df.columns) == 14


def test_make_train_test_datasets(mocker, tradingai_conf):
    tradingai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, tradingai_conf)
    tradingai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = tradingai.dk.filter_features(
            unfiltered_dataframe,
            tradingai.dk.training_features_list,
            tradingai.dk.label_list,
            training_filter=True,
        )

    data_dictionary = tradingai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    assert data_dictionary
    assert len(data_dictionary) == 7
    assert len(data_dictionary['train_features'].index) == 1916


@pytest.mark.parametrize('model', [
    'LightGBMRegressor'
    ])
def test_get_full_model_path(mocker, tradingai_conf, model):
    tradingai_conf.update({"tradingaimodel": model})
    tradingai_conf.update({"timerange": "20180110-20180130"})
    tradingai_conf.update({"strategy": "tradingai_test_strat"})

    if is_mac():
        pytest.skip("Mac is confused during this test for unknown reasons")

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

    model_path = tradingai.dk.get_full_models_path(tradingai_conf)
    assert model_path.is_dir() is True
