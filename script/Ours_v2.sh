#for debug
# python main.py -data NSLKDD -m CICIDS -algo ours -gr 50 -lbs 128 -nc 5 -t 2 -nb 5 -en_lr 0.1 -go our_test_beta_0_5
# python main.py -data NSLKDD_global -m CICIDS -algo ours -gr 50 -lbs 128 -nc 5 -t 1 -nb 5 -en_lr 0.1 -go our_test_global_beta_0_5
set -e
cd ../system

model=cicids # model name
nc=5 # number of client
t=1 #time
algo=ours
beta=0.5

use_personlized_agg=True
use_prototype_ensemble=False

go_normal=${algo}_client5_beta_${beta}_Ablation # expeiment description for normal test
go_global=${algo}_client5_GLOBAL_beta_${beta}_ablation # expeiment description for global test
#algo=ours
#########################################################
# mini_cicids_2018
gr=100 # global round
lbs=1024 # local batch size
nb=7 #number of classes

python main.py -data mini_cicids_2018_global_test -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_global} -s_c ${use_personlized_agg}
##########################################################
##NSL-KDD
#gr=50 # global round
#lbs=128 # local batch size
#nb=5 #number of classes
#
## NSL-KDD
#python main.py -data NSLKDD -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_normal}
### NSL-KDD (global_test)
#python main.py -data NSLKDD_global -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_global}
##
#
