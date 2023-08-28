# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for tradingai
"""
import logging
from pathlib import Path

from trading.constants import USERPATH_FREQAIMODELS, Config
from trading.exceptions import OperationalException
from trading.tradingai.tradingai_interface import IFreqaiModel
from trading.resolvers import IResolver


logger = logging.getLogger(__name__)


class FreqaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = IFreqaiModel
    object_type_str = "FreqaiModel"
    user_subdir = USERPATH_FREQAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("tradingai/prediction_models").resolve()
    )
    extra_path = "tradingaimodel_path"

    @staticmethod
    def load_tradingaimodel(config: Config) -> IFreqaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        tradingaimodel_name = config.get("tradingaimodel")
        if not tradingaimodel_name:
            raise OperationalException(
                "No tradingaimodel set. Please use `--tradingaimodel` to "
                "specify the FreqaiModel class to use.\n"
            )
        if tradingaimodel_name in disallowed_models:
            raise OperationalException(
                f"{tradingaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        tradingaimodel = FreqaiModelResolver.load_object(
            tradingaimodel_name,
            config,
            kwargs={"config": config},
        )

        return tradingaimodel
