# pragma pylint: disable=missing-docstring, protected-access, invalid-name

import pytest

from trading.enums import RunMode
from trading.exceptions import OperationalException
from trading.exchange.check_exchange import check_exchange
from tests.conftest import log_has_re


def test_check_exchange(default_conf, caplog) -> None:
    # Test an officially supported by Trading team exchange
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.get('exchange').update({'name': 'BITTREX'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is officially supported by the Trading development team\.",
                      caplog)
    caplog.clear()

    # Test an officially supported by Trading team exchange
    default_conf.get('exchange').update({'name': 'binance'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"binance\" is officially supported by the Trading development team\.",
        caplog)
    caplog.clear()

    # Test an officially supported by Trading team exchange
    default_conf.get('exchange').update({'name': 'binanceus'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"binanceus\" is officially supported by the Trading development team\.",
        caplog)
    caplog.clear()

    # Test an officially supported by Trading team exchange - with remapping
    default_conf.get('exchange').update({'name': 'okex'})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"Exchange \"okex\" is officially supported by the Trading development team\.",
        caplog)
    caplog.clear()
    # Test an available exchange, supported by ccxt
    default_conf.get('exchange').update({'name': 'huobipro'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is known to the the ccxt library, available for the bot, "
                      r"but not officially supported "
                      r"by the Trading development team\. .*", caplog)
    caplog.clear()

    # Test a 'bad' exchange, which known to have serious problems
    default_conf.get('exchange').update({'name': 'bitmex'})
    with pytest.raises(OperationalException,
                       match=r"Exchange .* will not work with Trading\..*"):
        check_exchange(default_conf)
    caplog.clear()

    # Test a 'bad' exchange with check_for_bad=False
    default_conf.get('exchange').update({'name': 'bitmex'})
    assert check_exchange(default_conf, False)
    assert log_has_re(r"Exchange .* is known to the the ccxt library, available for the bot, "
                      r"but not officially supported "
                      r"by the Trading development team\. .*", caplog)
    caplog.clear()

    # Test an invalid exchange
    default_conf.get('exchange').update({'name': 'unknown_exchange'})
    with pytest.raises(
        OperationalException,
        match=r'Exchange "unknown_exchange" is not known to the ccxt library '
              r'and therefore not available for the bot.*'
    ):
        check_exchange(default_conf)

    # Test no exchange...
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.PLOT
    assert check_exchange(default_conf)

    # Test no exchange...
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.UTIL_EXCHANGE
    with pytest.raises(OperationalException,
                       match=r'This command requires a configured exchange.*'):
        check_exchange(default_conf)
