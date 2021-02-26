#!/bin/sh

echo get pd...
#wget http://msp.ucsd.edu/Software/pd-0.47-1.src.tar.gz
#tar xvzf pd-0.47-1.src.tar.gz 
#mv  pd-0.47-1 pd


if [ ! -d pd/.git ];
then
 git clone git://git.code.sf.net/p/pure-data/pure-data pd
else 
 git -C pd pull
fi


cd pd
./autogen.sh 
./configure --enable-jack
make
echo --- done ---
