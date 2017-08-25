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

Výpis všech screenů:
````bash
screen -ls
````
