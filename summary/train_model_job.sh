export ABS=/nfs/isicvlnas01/users/sachinma/NAMAS
source /nas/home/sachinma/.bashrc
echo "Starting training for the model using the sample training set"
$ABS/train_model.sh $ABS/working_dir model_actual.th
exit $?
