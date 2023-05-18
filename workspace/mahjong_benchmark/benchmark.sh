echo "| function name | # expr lines | compile time |"
echo "| :--- | ---: | ---: |"
for funcname in can_riichi is_tenpai can_tsumo score number
do
    python3 benchmark.py $funcname
done
