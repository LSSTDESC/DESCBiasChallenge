#!/bin/bash
BMs='lin'
for BM in $BMs
do
    for kind in {1..1}
    do
        echo Bias model:${BM}
        kmax=`echo "($kind)*0.05" | bc -l`
        echo kmax=${kmax}
        #works python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/fid_red_HOD.fits --bias_model $BM --k_max $kmax --fit_params sigma8 cllike_cl1_b1 cllike_cl1_b1p --bins cl1 sh1 sh2 --path2output ./kmax_results/lin_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl1 sh2 sh2 sh2
        #also works python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/abacus_red_abacus.fits --bias_model $BM --k_max $kmax --fit_params sigma8 cllike_cl1_b1 cllike_cl1_b1p --bins cl1 sh1 sh2 --path2output ./kmax_results/aba_lin_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl1 sh2 sh2 sh2
        #works againpython kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/fid_red_HOD.fits --bias_model $BM --k_max $kmax --fit_params sigma8 Omega_c cllike_cl1_b1 cllike_cl1_b1p --bins cl1 sh1 sh2 --path2output ./kmax_results/hodomc_lin_${BM}_kmax_${kmax} --probes cl1 cl1 cl1 sh1 sh1 sh1 cl1 sh2 sh2 sh2
        python kmax_test.py --path2defaultconfig ./kmax_test.yml --path2data ../../data/fid_red_HOD.fits --bias_model $BM --k_max $kmax --fit_params sigma8 Omega_c cllike_cl1_b1 cllike_cl1_b1p cllike_cl2_b1 cllike_cl2_b1p cllike_cl3_b1 cllike_cl3_b1p cllike_cl4_b1 cllike_cl4_b1p cllike_cl5_b1 cllike_cl5_b1p cllike_cl6_b1 cllike_cl6_b1p --bins all --path2output ./kmax_results/all_hodomc_lin_${BM}_kmax_${kmax} --probes all
    done
echo Done $BM
done
echo All done
