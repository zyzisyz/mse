. ./path.sh

rm -f *_*_eer 
rm -f n 
rm -f log 
rm -f result

for b in 0.03;do
    for a in 15;do
        for (( n=0; n<=52; n+=3 ))
        do
            python -u eval.py   --epoch 350 \
                                --batch_size 200 \
                                --n_hidden 1800 \
                                --learn_rate 0.00005 \
                                --beta1 0.5 \
                                --dataset_path ./data/voxceleb_combined_200000/xvector.npz \
                                --z_dim 200 \
                                --a ${a} \
                                --n ${n} \
                                --b ${b}
            wait;
            
            echo $n>>n
            echo "---------------------">>result
            echo "n: " ${n}>>result
            echo "---------------------">>result

            for sub in eval dev; do
                # Cosine metric.
                
                echo "Test on SITW $sub:"
                echo "Test on SITW $sub:">>result
                local/cosine_scoring.sh data/sitw_$sub/enroll \
                                        data/sitw_$sub/test \
                                        data/sitw_$sub/test/core-core.lst \
                                        data/sitw_$sub/foo 
                eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
                echo "Cosine EER: $eer%"
                echo "Cosine EER: $eer%">>result
                echo $eer>>${sub}_cosine_eer
                
                
                # Create a PLDA model and do scoring.
                local/plda_scoring.sh data/voxceleb_combined_200000 \
                                        data/sitw_$sub/enroll \
                                        data/sitw_$sub/test \
                                        data/sitw_$sub/test/core-core.lst \
                                        data/sitw_$sub/foo 
                eer=$(paste data/sitw_$sub/test/core-core.lst data/sitw_$sub/foo/plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
                echo "PLDA EER: $eer%"
                echo "PLDA EER: $eer%">>result
                echo $eer>>${sub}_PLDA_eer

                echo >>result
            done
        done
    done
done