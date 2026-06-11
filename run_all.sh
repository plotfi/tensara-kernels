#!/bin/bash

for exe in build/bin/*.exe; do
    echo "==> $exe"
    "$exe"
    echo ""
done
