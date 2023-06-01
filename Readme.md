Reshift
=======

Repeated Pitch-shifting for Frequency Discretization as Artistic Effect for Speech Signals.

This repository originated during the course of the audio engineering project by Manuel Planton at the institute for electronic music and acoustics in Graz.
Extensive documentation about the research and implementation is found [here](Audio_Engineering_Project_-_Manuel_Planton.pdf).

The goal was to develop a real-time capable pitch discretization effect similar to *Autotune* and explore artistic effects by repeatedly applying it to a speech signal.

The resulting artistic experiments are in "pd/experiment.pd" where pitch-discretization effects are used which use two different pitch-shifting algorithms (granular, "Rollers").

In this project, pitch-tracking and pitch-shifting algorithms had been explored in Jupyter Notebooks, which lead to an implementation of the pitch-discretization effect called "Reshift" using the "Rollers" pitch-shifting algorithm.
The realtime implementation of two pitch discretization effects, the experiments for repeatedly applying pitch-shifting and a collection of different audio effects using pitch-discretization are implemented in pure data.


Directories
-----------

* notebooks: Jupyter Notebooks for developing algorithms and testing

* py: implementation of the developed algorithms

* pd: realtime implementation of two pitch-discretization effects, experiments and different audio effects using pitch-discretization


Pitch-Discretization
--------------------

A pitch discretization effect consists of

* a pitch-tracking algorithm

* a nonlinear frequency scale for the target pitch

* and of a pitch-shifting algorithm.


The developed realtime effects for pitch-discretization are:

* pd/reshift~.pd: the Reshift pitch-discretization effect using Rollers pitch-shifting

* pd/disco~.pd: a pitch discretization effect using a granular method for pitch-shifting


Notebooks
---------

These jupyter notebooks document the development, exploration and testing of the algorithms used in this project.
See the [index](notebooks/Index.ipynb) for an overview of the notebooks.


Open Jupyter Notebooks
----------------------

```
conda activate base
jupyter notebook
```

Then select the notebook you want to open in your browser.


Installation
------------

__pure data environment:__

```
sudo apt install gcc make linux-headers-$(uname -r) build-essential automake autoconf libtool gettext git libjack-jackd2-dev libasound2-dev
sudo apt install libaubio-dev
pd/make_pd.sh
pd/make_libs.sh
```

__python environment:__

* get the anaconda installer for your system from [here](https://www.anaconda.com/products/individual)

* install anaconda

* optionally deactivate conda's base environment with
        conda config --set auto_activate_base false

* install dependencies

```
conda activate base
pip3 install presets librosa snakeviz
```


Possible Audio Effects Using Pitch-discretization with Rollers Pitch-shifting:
------------------------------------------------------------------------------

* pd/FX1.pd: audio effects using one instance of Reshift

    - discrete frequency shifter (using one band in Rollers)
    
    - pitch discretization (just the Reshift effect using 200 bands)
    
    - inharmonizer (individual voices)
    
    - vibrato
    
    - two voices
    
    - scramble
    
    - delay with Rollers in dry feed forward path
    
    - delay with Rollers in wet feed forward path
    
    - delay with Rolers in feedback path

* pd/FX2.pd: audio effects using two instances of Reshift

    - two discretized voices
    
    - three voices (two discretized and original)
    
    - vibratos: two out of phase vibrato effects
    
    - pitch cancellation: pitch shift up and same pitch shift down causing artifacts
    
    - delay with discretized pitch shifting dry and wet feed forward paths
    
    - delay with discretized pitch shifting in feed forward and feedback path

* pd/FX_many.pd: audio effects using many instances of Reshift

    - pitch cancellation: cancelling pitch shifting with 4 Reshift instances
    
    - pitch discretized chorus
    
    - inharmonizer with parallel voices
    
    - harmonizer: seventh chord

