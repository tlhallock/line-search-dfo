#!/bin/bash
find -name '*.pyomo.nl' -delete
find -name '*.pyomo.sol' -delete
find images -name '*.png' -delete
find images -name '*_debug_output.txt' -delete
find images -name 'different_ellipses_*' -delete
rm -f images/evaluations.txt
find images -name 'log.txt' -delete


































