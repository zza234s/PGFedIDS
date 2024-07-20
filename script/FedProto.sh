set -e
cd ../system

algo=FedProto
model=cicids # model name
nc=5 # number of client
t=1 #time
beta=0.5
go_normal=beta_${beta} # expeiment description for normal test
go_global=beta_${beta}_global # expeiment description for global test


#########################################################
# mini_cicids_2018
gr=100 # global round
lbs=1024 # local batch size
nb=7 #number of classes

# mini_cicids_2018
#python main.py -data mini_cicids_2018 -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_normal}
# mini_cicids_2018 (global_test)
python main.py -data mini_cicids_2018_global_test -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_global}

#########################################################
#NSL-KDD
gr=50 # global round
lbs=128 # local batch size
nb=5 #number of classes

# NSL-KDD
#python main.py -data NSLKDD -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_normal}
# NSL-KDD (global_test)
#python main.py -data NSLKDD_global -m ${model} -algo ${algo} -gr ${gr} -lbs ${lbs} -nc ${nc} -t ${t} -nb ${nb} -go ${go_global}
