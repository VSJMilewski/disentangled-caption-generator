#PBS -qgpu
#PBS -lnodes=1
#PBS -lwalltime=48:00:00

module load eb/3.7.0
module load python/3.5.0
module load cuDNN/7.3.1-CUDA-9.0.176


echo "creating a tar for the data and pickles"
tar zcf $HOME/multimodal-descriptions/data.tar.gz -C $HOME/multimodal-descriptions/ data
#echo "moving the pickles to scratch"
cp -r $HOME/multimodal-descriptions/pickles/ "$TMPDIR"
echo "untar the data into scratch"
tar zxf $HOME/multimodal-descriptions/data.tar.gz -C "$TMPDIR"
echo "make output dir"
mkdir "$TMPDIR"/output

echo "move into scratch"
cd "$TMPDIR"
ls -l

echo "run program..."
echo ""

python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --model BINARY --dataset flickr8k --device cuda --num_workers 16 --batch_size 256 --min_epochs 0 --max_epochs 1000 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 5 --optimizer Adam --dropout_prob 0.1 --binary_train_method WEIGHTED
python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --model BINARY --dataset flickr8k --device cuda --num_workers 16 --batch_size 256 --min_epochs 0 --max_epochs 1000 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 5 --optimizer Adam --dropout_prob 0.2 --binary_train_method WEIGHTED
python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --model BINARY --dataset flickr8k --device cuda --num_workers 16 --batch_size 256 --min_epochs 0 --max_epochs 1000 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 5 --optimizer Adam --dropout_prob 0.3 --binary_train_method WEIGHTED
python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --model BINARY --dataset flickr8k --device cuda --num_workers 16 --batch_size 256 --min_epochs 0 --max_epochs 1000 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 5 --optimizer Adam --dropout_prob 0.4 --binary_train_method WEIGHTED
python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --model BINARY --dataset flickr8k --device cuda --num_workers 16 --batch_size 256 --min_epochs 0 --max_epochs 1000 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 5 --optimizer Adam --dropout_prob 0.9 --binary_train_method WEIGHTED

echo ""
echo ""
echo "tarring output to home"
tar zcf $HOME/multimodal-descriptions/output_$PBS_JOBID.tar.gz output pickles
echo "Job $PBS_JOBID, dropout tuner bin_w ended at `date`" | mail $USER -s "Job $PBS_JOBID"