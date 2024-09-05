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
sed -i '/add_library(metis ${METIS_LIBRARY_TYPE} ${metis_sources})/ s/$/\ntarget_link_libraries(metis GKlib)/' libmetis/CMakeLists.txt
