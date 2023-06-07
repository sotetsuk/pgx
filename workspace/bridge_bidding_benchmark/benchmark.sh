echo "| function name | # expr lines | compile time |"
echo "| :--- | ---: | ---: |"
for funcname in _init_by_key _step _observe _key_to_hand _shuffle_players _state_to_key _to_binary duplicate _terminated_step _continue_step _state_bid _find_value_from_key
do
    python3 benchmark.py $funcname
done
