#!/bin/bash
BMs='anzu'
for BM in $BMs
do
    for kind in {1..1}
    do
        echo Bias model:${BM}
        kmax=`echo "($kind)*0.05" | bc -l`
        echo kmax=${kmax}
        python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/fid_red_HOD.fits --bias_model $BM --k_max $kmax --fit_params sigma8 Omega_c cllike_cl1_b1 cllike_cl1_b1p cllike_cl2_b1 cllike_cl2_b1p cllike_cl3_b1 cllike_cl3_b1p cllike_cl4_b1 cllike_cl4_b1p cllike_cl5_b1 cllike_cl5_b1p cllike_cl6_b1 cllike_cl6_b1p --bins cl1 cl2 cl3 cl4 cl5 cl6 --path2output ./kmax_results/pkanzu_hodomc_lin_${BM}_kmax_${kmax} --probes cl1 cl1 cl2 cl2 cl3 cl3 cl4 cl4 cl5 cl5 cl6 cl6
    done
echo Done $BM
done
echo All done
