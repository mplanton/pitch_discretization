#!/bin/sh

echo get libaubio...

if [ ! -d pd-aubio/.git ];
then
    git clone https://github.com/aubio/pd-aubio.git
else 
    git -C pd-aubio pull
fi

cd pd-aubio
./waf configure build
make

echo --- done ---
