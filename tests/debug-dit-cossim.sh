#!/bin/bash

./debug-dit-cossim.py --mode both --quant BF16 > BF16.log
./debug-dit-cossim.py --mode both --quant Q8_0 > Q8_0.log
./debug-dit-cossim.py --mode both --quant Q6_K > Q6_K.log
./debug-dit-cossim.py --mode both --quant Q5_K_M > Q5_K_M.log
