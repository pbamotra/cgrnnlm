make -j5
e=$?
basedir='../benchmarks'
data=$basedir/simple-examples/data
hidden_size=100
context_size=10
mname="faster-rnnlm"
rm -rf $basedir/models
mkdir -p $basedir/models
retry=0
context_loss_weight=0
if [[ $e -eq 0 ]]
then
    exe=./rnnlm 
 $exe -rnnlm $basedir/models/$mname -train $data/ptb.train.txt -valid $data/ptb.valid.txt -hidden $hidden_size --context_size $context_size --retry $retry -threads 1 -context_loss_weight $context_loss_weight  2>&1 | tee f.out
 #$exe -rnnlm $basedir/models/$2 -test $data/ptb.test.txt -nce-accurate-test 1 2>&1 > /dev/null | grep     "Test entropy" | cat
fi
