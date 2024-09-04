mkdir lib
cd lib
git clone https://github.com/KarypisLab/GKlib.git
git clone https://github.com/KarypisLab/METIS.git
cd GKlib
make config CONFIG_FLAGS='-D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=~/local'
make install
cd ../METIS
make config shared=1 cc=gcc prefix=~/local
make install