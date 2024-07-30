# If you are using a linux distribution
# you can use this script to install the necessary dependencies to run the project.
# Check the gfootball/doc/compile_engine.md file for more information.
# after you clone it:
# git clone https://github.com/Gabrynho/RLproject_football.git

sudo apt-get update

sudo apt-get install -y git cmake make build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip

python3 -m pip install --upgrade pip
pip install setuptools==65.5.0
pip install wheel==0.38.0
python3 -m pip install psutil

#create virtual environment with conda
conda create --name football-env python=3.9 -y
conda activate football-env

python3 -m pip install --upgrade pip
pip install setuptools==65.5.0
pip install wheel==0.38.0
python3 -m pip install psutil

conda install libgcc==5.2.0 -y
conda install -c conda-forge gcc=12.1.0 -y


cd RLproject_football && python3 -m pip install .