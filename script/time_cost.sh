
gr=100 # global round
lbs=1024 # local batch size
nb=7 #number of classes
model=cicids

# FedAvg
python main.py -data mini_cicids_2018_global_test -m cicids -algo FedAvg -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go FedAvg_time_exp

# Local
python main.py -data mini_cicids_2018_global_test -m cicids -algo Local -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go Local_time_exp

# FedProto
python main.py -data mini_cicids_2018_global_test -m cicids -algo FedProto -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go FedProto_time_exp

#FedGH
python main.py -data mini_cicids_2018_global_test -m cicids -algo FedGH -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go FedGH_time_exp

#MOON
python main.py -data mini_cicids_2018_global_test -m cicids -algo MOON -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go MOON_time_exp

#GPFL
python main.py -data mini_cicids_2018_global_test -m cicids -algo GPFL -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go GPFL_time_exp

#ours
python main.py -data mini_cicids_2018_global_test -m cicids -algo ours -gr 100 -lbs 1024  -nc 5 -t 1 -nb 7 -go ours_time_exp