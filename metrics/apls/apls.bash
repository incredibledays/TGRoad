declare -a arr=(8 9 19 28 29 39 48 49 59 68 69 79 88 89 99 108 109 119 128 129 139 148 149 159 168 169 179)
dir=../../results/exp2
echo $dir
mkdir $dir/apls
for i in "${arr[@]}"
do
    if test -f "${dir}/region_${i}_graph.p"; then
        echo "========================$i======================"
        python ./convert.py "../../datasets/cityscale/test/region_${i}_refine_gt_graph.p" gt.json
        python ./convert.py "${dir}/region_${i}_graph.p" prop.json
        go run ./main.go gt.json prop.json "${dir}/apls/${i}.txt"
    fi
done
python apls.py --dir $dir