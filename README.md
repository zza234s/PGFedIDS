# Readme

The official code for the paper *Prototype Guided Personalized Federated Intrusion Detection System.*  

All baselines and our method are implemented on the PFLlib. We are very grateful for this outstanding library.



## Dataset

Please download all used datasets [here](https://drive.google.com/file/d/1mS2fbBCeXSvNeOlrvd0sOme2uUgTKqpJ/view?usp=sharing)

For example, to set up the NSLKDD dataset ($\beta=1.0$) for personalized testing, follow these steps:

- Unzip the Dataset.zip file.
- Navigate to the unzipped folder: "beta_1.0/NSLKDD".
- Copy the "train" and "test" folders into the "dataset/NSLKDD_global" folder in the source code repository.



## Quickly Start (PG-FedIDS)

```shell
cd system 

# NSLKDD (Personalized test)
python main.py -data NSLKDD -m cicids -algo ours -gr 128 -lbs 1024 -nc 5 -nb 5

# NSLKDD (Global test)
python main.py -data NSLKDD_global -m cicids -algo ours -gr 50 -lbs 128 -nc 5  -nb 5

# CICIDS2018 (Personalized test)
python main.py -data mini_cicids_2018 -m cicids -algo ours -gr 100 -lbs 1024 -nc 5 -nb 7

# CICIDS2018 (Global test)
python main.py -data mini_cicids_2018_global_test -m cicids -algo ours -gr 100 -lbs 1024 -nc 5 -nb 7
```



## Run Baselines 

Users can run all baselines similarly to running PG-FedIDS by adjusting the `-algo`  parameter with the desired baseline name. 

For instance, to run baselines on the NSLKDD dataset under global testing:

```shell
cd system

#Local
python main.py -data NSLKDD_global -m cicids -algo Local -gr 50 -lbs 128 -nc 5  -nb 5

#MOON
python main.py -data NSLKDD_global -m cicids -algo MOON -gr 50 -lbs 128 -nc 5  -nb 5

#GPFL
python main.py -data NSLKDD_global -m cicids -algo GPFL -gr 50 -lbs 128 -nc 5  -nb 5

#FedProto
python main.py -data NSLKDD_global -m cicids -algo FedProto -gr 50 -lbs 128 -nc 5  -nb 5

#FedGH
python main.py -data NSLKDD_global -m cicids -algo FedGH -gr 50 -lbs 128 -nc 5  -nb 5
```



