echo "| function name | # jaxpr lines | compile time |"
echo "| :--- | ---: | ---: |"
# for funcname in init step _dlaction_to_action _action_to_dlaction _piece_moves _legal_actions _init_legal_actions _is_mate
for funcname in _action_to_dlaction _piece_moves _legal_actions _init_legal_actions _is_mate
do
    python3 benchmark.py $funcname
done
