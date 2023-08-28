ARG sourceimage=tradingorg/trading
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /trading/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
