#PBS -qgpu
#PBS -lnodes=1
#PBS -lwalltime=48:00:00

echo "creating a tar for the data and pickles"
tar zcf $HOME/multimodal-descriptions/data.tar.gz -C $HOME/multimodal-descriptions/ data pickles
#echo "moving the pickles to scratch"
#cp $HOME/multimodal-descriptions/*.pkl "$TMPDIR"
echo "untar the data into scratch"
tar zxf $HOME/multimodal-descriptions/data.tar.gz -C "$TMPDIR"
echo "make output dir"
mkdir "$TMPDIR"/output

echo "move into scratch"
cd "$TMPDIR"
ls -l

echo "run program..."
echo ""

python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr8k --device cuda:0 &
python3 $HOME/multimodal-descriptions/main.py --beam_size 20 --dataset flickr8k --device cuda:1 &
python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr30k --device cuda:2 &
python3 $HOME/multimodal-descriptions/main.py --beam_size 20 --dataset flickr30k --device cuda:3 &

echo ""
echo ""
echo "tarring output to home"
tar zcf $HOME/multimodal-descriptions/output_$PBS_JOBID.tar.gz output
