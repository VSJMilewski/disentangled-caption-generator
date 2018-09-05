#PBS -qgpu
#PBS -lnodes=1
#PBS -lwalltime=00:05:00

tar zcvf $HOME/multimodal-descriptions/data.tar.gz $HOME/multimodal-descriptions/data
cd "$TMPDIR"
cp $HOME/multimodal-descriptions/*.pkl "$TMPDIR"
tar zxf $HOME/multimodal-descriptions/data.tar.gz
mkdir "$TMPDIR"/output


python3 $HOME/multimodal-descriptions/main.py

tar zcf $HOME/multimodal-descriptions/output_$HOSTNAME.tar.gz output