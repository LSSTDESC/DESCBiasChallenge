#!/bin/bash
BMs='lin EuPT LPT'
for BM in $BMs
do
    for kind in {1..8}
    do
        echo Bias model:${BM}
        kmax=`echo "($kind)*0.05" | bc -l`
        echo kmax=${kmax}
        python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/abacus_red_abacus.fits --bias_model $BM --k_max $kmax --fit_params sigma8 --bins cl1 cl2 --path2output ./kmax_results/bm_${BM}_kmax_${kmax} --probes cl1 cl2
    done
echo Done $BM
done
echo All done
