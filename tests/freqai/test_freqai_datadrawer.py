
import shutil
from pathlib import Path

import pytest

from trading.configuration import TimeRange
from trading.data.dataprovider import DataProvider
from trading.exceptions import OperationalException
from trading.tradingai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.tradingai.conftest import get_patched_tradingai_strategy


def test_update_historic_data(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")

    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    historic_candles = len(tradingai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    tradingai.dk.pair = "ADA/BTC"
    tradingai.dd.update_historic_data(strategy, tradingai.dk)

    updated_historic_candles = len(tradingai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_load_all_pairs_histories(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)

    assert len(tradingai.dd.historic_data.keys()) == len(
        tradingai_conf.get("exchange", {}).get("pair_whitelist")
    )
    assert len(tradingai.dd.historic_data["ADA/BTC"]) == len(
        tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).get("include_timeframes")
    )
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_get_base_and_corr_dataframes(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)

    num_tfs = len(
        tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).get("include_timeframes")
    )

    assert len(base_df.keys()) == num_tfs

    assert len(corr_df.keys()) == len(
        tradingai_conf.get("tradingai", {}).get("feature_parameters", {}).get("include_corr_pairlist")
    )

    assert len(corr_df["ADA/BTC"].keys()) == num_tfs
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_use_strategy_to_populate_indicators(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    strategy.tradingai_info = tradingai_conf.get("tradingai", {})
    tradingai = strategy.tradingai
    tradingai.live = True
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", tradingai.dk)

    df = tradingai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')

    assert len(df.columns) == 33
    shutil.rmtree(Path(tradingai.dk.full_path))


def test_get_timerange_from_live_historic_predictions(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    tradingai = strategy.tradingai
    tradingai.live = False
    tradingai.dk = FreqaiDataKitchen(tradingai_conf)
    tradingai.dk.live = False
    timerange = TimeRange.parse_timerange("20180126-20180130")
    tradingai.dd.load_all_pair_histories(timerange, tradingai.dk)
    sub_timerange = TimeRange.parse_timerange("20180128-20180130")
    _, base_df = tradingai.dd.get_base_and_corr_dataframes(sub_timerange, "ADA/BTC", tradingai.dk)
    base_df["5m"]["date_pred"] = base_df["5m"]["date"]
    tradingai.dd.historic_predictions = {}
    tradingai.dd.historic_predictions["ADA/USDT"] = base_df["5m"]
    tradingai.dd.save_historic_predictions_to_disk()
    tradingai.dd.save_global_metadata_to_disk({"start_dry_live_date": 1516406400})

    timerange = tradingai.dd.get_timerange_from_live_historic_predictions()
    assert timerange.startts == 1516406400
    assert timerange.stopts == 1517356500


def test_get_timerange_from_backtesting_live_df_pred_not_found(mocker, tradingai_conf):
    strategy = get_patched_tradingai_strategy(mocker, tradingai_conf)
    exchange = get_patched_exchange(mocker, tradingai_conf)
    strategy.dp = DataProvider(tradingai_conf, exchange)
    tradingai = strategy.tradingai
    with pytest.raises(
            OperationalException,
            match=r'Historic predictions not found.*'
            ):
        tradingai.dd.get_timerange_from_live_historic_predictions()
