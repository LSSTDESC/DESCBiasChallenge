#!/bin/bash
BMs='EuPT'
for BM in $BMs
do
    for kind in {1..1}
    do
        echo Bias model:${BM}
        kmax=`echo "($kind)*0.151" | bc -l`
        echo kmax=${kmax}
        python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/fid_red_HOD.fits --bias_model $BM --k_max $kmax --fit_params sigma8 cllike_cl1_b1 cllike_cl1_b1p cllike_cl1_b2 cllike_cl1_b2 cllike_cl1_bsn --bins cl1 sh1 sh2 --path2output ./kmax_results/1tbm_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl1 sh2 sh2 sh2
        #python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/abacus_red_abacus.fits --bias_model $BM --k_max $kmax --fit_params sigma8 cllike_cl1_b1 cllike_cl6_b1 cllike_cl1_b1p cllike_cl6_b1p cllike_cl1_b2 cllike_cl6_b2 cllike_cl1_bs cllike_cl6_bs cllike_cl1_bsn cllike_cl6_bsn --bins cl1 sh1 cl6 sh2 --path2output ./kmax_results/1tbm_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl6 cl6 cl6 sh1 cl1 sh2 cl6 sh2 sh2 sh2
    done
echo Done $BM
done
echo All done
