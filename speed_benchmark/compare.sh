echo "| library | game | type | n_envs | steps | time | time per step |"
echo "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
python compare_subproc.py open_spiel go 10 1000
python compare_subproc.py petting_zoo go 10 1000
python compare_for_loop.py open_spiel go 10 1000
python compare_for_loop.py petting_zoo go 10 1000