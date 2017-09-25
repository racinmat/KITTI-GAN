### how to install things on server
Věci jde instalovat jen přes virtual env. 
Defaulně se mi aktivuje díky úpravě .bashrc:

````bash
if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH="/home.stud/racinmat/GANy"
else
  export PYTHONPATH="${PYTHONPATH}:/home.stud/racinmat/GANy"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

source gan2/bin/activate
```` 

Při startování dlouho běžících tasků je třeba použít screen, to mám automatizované ve sktiptech:
run-python-bg.sh:
````bash
#!/bin/bash

screen -dmSL python ./run-python.sh $1
````

run-tensorboard-bg.sh:
````bash
#!/bin/bash

screen -dmSL tensorboard ./run-tensorboard.sh $1
````
run-tensorboard.sh
````bash
#!/bin/bash
if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH="/home.stud/racinmat/GANy"
else
  export PYTHONPATH="${PYTHONPATH}:/home.stud/racinmat/GANy"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

source gan2/bin/activate

tensorboard --logdir=./GANy/logs/$1
````

runpython.sh
````bash
#!/bin/bash
if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH="/home.stud/racinmat/GANy"
else
  export PYTHONPATH="${PYTHONPATH}:/home.stud/racinmat/GANy"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

source gan2/bin/activate

cd GANy
python python/neural_network/$1
````

Když potřebuju task zabít, tak se připojím na screen:
````bash
screen -r [název screenu]
````
a pak mohu CTRL+C zabít daný task a opustit screen, tím zmizí.

Odpojení se od obrazovky po připojení přes -r
````bash
ctrl+a d
````


Výpis všech screenů:
````bash
screen -ls
````

Velikost složek a souborů do hloubky 3:
````bash
du -h --max-depth=3
````


Vypíše md5 chcecksum složky
````bash
find ./logs/1503829581 -type f -exec md5sum {} \; | sort -k 2 | md5sum

````

Sledování výkonu GPU:
````bash
watch -n 2 nvidia-smi
````

volné místo na disku:
````bash
df -h
````

spuštění učení s parametry:
````bash
./run-python-bg.sh train_gan_photos.py --output_dir=/datagrid/personal/racinmat --type=dropouts
./run-python-bg.sh train_gan_photos.py --output_dir=/datagrid/personal/racinmat --type=dropouts --l1_ratio=50 --epoch=600
````


spuštění tensorboardu s absolutní cestou:
````bash
./run-tensorboard-bg.sh /datagrid/personal/racinmat/log
````