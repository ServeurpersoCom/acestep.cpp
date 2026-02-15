#!/bin/bash

set -eu

for script in *.sh; do
    [ "$script" = "all.sh" ] && continue
    bash "$script"
done
