from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from trading.commands.optimize_commands import setup_optimize_configuration
from trading.enums import RunMode
from trading.exceptions import OperationalException
from trading.optimize.backtesting import Backtesting
from tests.conftest import (CURRENT_TEST_STRATEGY, get_args, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


def test_tradingai_backtest_start_backtest_list(tradingai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('trading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('trading.optimize.backtesting.history.load_data')
    mocker.patch('trading.optimize.backtesting.history.get_timerange', return_value=(now, now))

    patched_configuration_load_config_file(mocker, tradingai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--strategy-list', CURRENT_TEST_STRATEGY
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re('Using --strategy-list with TradingAI REQUIRES all strategies to have identical',
                      caplog)
    Backtesting.cleanup()


def test_tradingai_backtest_load_data(tradingai_conf, mocker, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('trading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('trading.optimize.backtesting.history.load_data')
    mocker.patch('trading.optimize.backtesting.history.get_timerange', return_value=(now, now))
    backtesting = Backtesting(deepcopy(tradingai_conf))
    backtesting.load_bt_data()

    assert log_has_re('Increasing startup_candle_count for tradingai to.*', caplog)

    Backtesting.cleanup()


def test_tradingai_backtest_live_models_model_not_found(tradingai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('trading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('trading.optimize.backtesting.history.load_data')
    mocker.patch('trading.optimize.backtesting.history.get_timerange', return_value=(now, now))
    tradingai_conf["timerange"] = ""
    tradingai_conf.get("tradingai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, tradingai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--tradingai-backtest-live-models'
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(OperationalException,
                       match=r".* Historic predictions data is required to run backtest .*"):
        Backtesting(bt_config)

    Backtesting.cleanup()
