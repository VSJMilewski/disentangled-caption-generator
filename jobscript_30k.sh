#PBS -qgpu
#PBS -lnodes=1
#PBS -lwalltime=3:00:00

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

python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr30k --device cuda --num_workers 4 --batch_size 1024 --min_epochs 0 --max_epochs 500 --patience 10 --vocab_threshold 5 --max_time 9000 --max_grad 10 --optimizer Adagrad --unique adagrad_gradclip10_2lstm &
#python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr8k --device cuda:1 --min_epochs 50 --patience 10 --vocab_threshold 5 --max_time 169200 --max_grad 10 --optimizer Adam --unique adam_grad10 &
#python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr8k --device cuda:2 --min_epochs 50 --patience 10 --vocab_threshold 5 --max_time 169200 --optimizer Adagrad --unique adagrad &
#python3 $HOME/multimodal-descriptions/main.py --beam_size 1 --dataset flickr8k --device cuda:3 --min_epochs 50 --patience 10 --vocab_threshold 5 --max_time 169200 --optimizer RMSprop --unique rmsprop &
wait

echo ""
echo ""
echo "tarring output to home"
tar zcf $HOME/multimodal-descriptions/output_$PBS_JOBID.tar.gz output pickles
echo "Job $PBS_JOBID with flickr8k ended at `date`" | mail $USER -s "Job $PBS_JOBID"