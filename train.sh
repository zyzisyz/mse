          
python -u main.py   --epoch 20 \
                    --batch_size 200 \
                    --n_hidden 1800 \
                    --learn_rate 0.000005 \
                    --beta1 0.5 \
                    --dataset_path ./data/voxceleb_combined_200000/xvector.npz \
                    --z_dim 200 
