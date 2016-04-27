make -j5
e=$?
basedir='../benchmarks'
data=$basedir/simple-examples/data
hidden_size=40
context_size=10
context_loss=0
mname="faster-rnnlm"
rm -rf $basedir/models
mkdir -p $basedir/models
retry=0
if [[ $e -eq 0 ]]
then
    exe=./rnnlm 
 $exe -train_and_test 1 -rnnlm $basedir/models/$mname -train $data/ptb.train.txt -valid $data/ptb.valid.txt -hidden $hidden_size --context_size $context_size --retry $retry -threads 1 -context_loss_weight $context_loss -test $data/ptb.test.txt -nce-accurate-test 1 2>&1 | tee f.out
fi
