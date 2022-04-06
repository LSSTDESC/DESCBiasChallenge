#!/bin/bash
BMs='HZPT'
for BM in $BMs
do
    for kind in {1..1}
    do
        echo Bias model:${BM}
        kmax=`echo "($kind)*0.05" | bc -l`
        echo kmax=${kmax}
        python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/abacus_red_abacus.fits --bias_model $BM --k_max $kmax --fit_params sigma8 Omega_c cllike_cl1_b1 cllike_cl1_sngg cllike_cl1_A0gg cllike_cl1_Rgg cllike_cl1_R1hgg cllike_cl1_A0gm cllike_cl1_Rgm cllike_cl1_R1hgm --bins cl1 sh1 sh2 --path2output ./kmax_results/11_sh12hzpt_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl1 sh2 sh2 sh2 sh1 sh2
    done
echo Done $BM
done
echo All done
