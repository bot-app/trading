#!/bin/sh

# The below assumes a correctly setup docker buildx environment

IMAGE_NAME=bot-app/trading
CACHE_IMAGE=bot-app/trading_cache
# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
TAG_FREQAI=${TAG}_tradingai
TAG_FREQAI_RL=${TAG_FREQAI}rl
TAG_PI="${TAG}_pi"

PI_PLATFORM="linux/arm/v7"
echo "Running for ${TAG}"
CACHE_TAG=${CACHE_IMAGE}:${TAG_PI}_cache

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > trading_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    # Build regular image
    docker build -t trading:${TAG} .
    # Build PI image
    docker buildx build \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Build regular image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t trading:${TAG} .

    # Pull last build to avoid rebuilding the whole image
    # docker pull --platform ${PI_PLATFORM} ${IMAGE_NAME}:${TAG}
    # disable provenance due to https://github.com/docker/buildx/issues/1509
    docker buildx build \
        --cache-from=type=registry,ref=${CACHE_TAG} \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
fi

if [ $? -ne 0 ]; then
    echo "failed building multiarch images"
    return 1
fi
# Tag image for upload and next build step
docker tag trading:$TAG ${CACHE_IMAGE}:$TAG

docker build --build-arg sourceimage=trading --build-arg sourcetag=${TAG} -t trading:${TAG_PLOT} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=trading --build-arg sourcetag=${TAG} -t trading:${TAG_FREQAI} -f docker/Dockerfile.tradingai .
docker build --build-arg sourceimage=trading --build-arg sourcetag=${TAG_FREQAI} -t trading:${TAG_FREQAI_RL} -f docker/Dockerfile.tradingai_rl .

docker tag trading:$TAG_PLOT ${CACHE_IMAGE}:$TAG_PLOT
docker tag trading:$TAG_FREQAI ${CACHE_IMAGE}:$TAG_FREQAI
docker tag trading:$TAG_FREQAI_RL ${CACHE_IMAGE}:$TAG_FREQAI_RL

# Run backtest
docker run --rm -v $(pwd)/config_examples/config_bittrex.example.json:/trading/config.json:ro -v $(pwd)/tests:/tests trading:${TAG} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

docker images

docker push ${CACHE_IMAGE}:$TAG
docker push ${CACHE_IMAGE}:$TAG_PLOT
docker push ${CACHE_IMAGE}:$TAG_FREQAI
docker push ${CACHE_IMAGE}:$TAG_FREQAI_RL

docker images

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi
