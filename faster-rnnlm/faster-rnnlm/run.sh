make -j5
e=$?
basedir='../benchmarks'
data=$basedir/simple-examples/data
hidden_size=40
context_size=10
context_loss=0
normalize=1
l1_loss=1
beta_file="../../pbamotra_data/data_3/lda_betas.csv"
dict_file="../../pbamotra_data/data_3/dictionary.ssv"
mname="faster-rnnlm"
rm -rf $basedir/models
mkdir -p $basedir/models
retry=0
if [[ $e -eq 0 ]]
then
    exe=./rnnlm 
 $exe -train_and_test 1 -rnnlm $basedir/models/$mname -train $data/ptb.train.txt -valid $data/ptb.valid.txt -hidden $hidden_size --context_size $context_size --retry $retry -threads 1 -context_loss_weight $context_loss -test $data/ptb.test.txt -nce-accurate-test 1 -normalize $normalize -l1_loss $l1_loss -beta_filepath $beta_file -dict_filepath $dict_file 2>&1 | tee f.out
fi
