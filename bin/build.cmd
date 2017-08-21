docker build -f ./docker/Dockerfile-extract-smop -t azathoth/extract-smop ./docker
docker build -f ./docker/Dockerfile-extract -t azathoth/extract ./docker
docker build -f ./docker/Dockerfile-gan -t azathoth/gan ./docker
docker build -f ./docker/Dockerfile-pyall -t azathoth/pyall ./docker
