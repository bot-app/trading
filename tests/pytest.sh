#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=trading --cov-config=.coveragerc tests/
