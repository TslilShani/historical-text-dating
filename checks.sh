#!/usr/bin/env bash

set -u -e -o pipefail

inv formatter --fix

set +e +u