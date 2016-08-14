#!/bin/bash

set -u
set -e

VEC=${1:-text8.tar.gz}
DIR=${2:-word-similarities/}
set +e; shift; shift; set -e

# from http://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Evaluating $VEC on word rankings in $DIR" >&2

python "$SCRIPTDIR/evalrank.py" "$@" -q "$VEC" `find "$DIR" -name '*.txt'`
