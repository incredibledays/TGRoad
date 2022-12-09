dir=../../results/exp2
mkdir $dir/topo

python ./main.py -savedir $dir
python ./topo_avg.py -savedir $dir