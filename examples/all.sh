#!/bin/bash

set -eu

for script in *.sh; do
    [ "$script" = "all.sh" ] && continue
    ./"$script"
done
