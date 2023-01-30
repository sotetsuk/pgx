echo "| function name | # expr lines | compile time |"
echo "| :--- | ---: | ---: |"
for funcname in step _update_state_wo_legal_action _pass_move _not_pass_move _merge_ren _set_stone_next_to_oppo_ren _remove_stones legal_actions _get_reward _count_ji _get_alphazero_features
do
    python3 benchmark.py $funcname
done
