make -j5
e=$?
basedir='../benchmarks'
data=$basedir/simple-examples/data
hidden_size=148
context_size=20
context_loss=0.01
pos_context_size=45
normalize=1
l1_loss=1
beta_file="../../pbamotra_data/data_6/lda_betas.csv"
dict_file="../../pbamotra_data/data_6/dictionary.ssv"
pos_file="../../pbamotra_data/data_6/pt_mat.csv"
mname="faster-rnnlm"
rm -rf $basedir/models
mkdir -p $basedir/models
retry=10
if [[ $e -eq 0 ]]
then
    exe=./rnnlm 
 $exe -train_and_test 1 -rnnlm $basedir/models/$mname -train $data/ptb.train.txt -valid $data/ptb.valid.txt -bptt 10 -nce 20 -hidden $hidden_size --context_size $context_size --pos_context_size $pos_context_size --retry $retry -threads 2 -context_loss_weight $context_loss -test $data/ptb.test.txt -nce-accurate-test 1 -normalize $normalize -l1_loss $l1_loss -beta_filepath $beta_file -pos_filepath $pos_file -dict_filepath $dict_file 2>&1 | tee f.out
fi
