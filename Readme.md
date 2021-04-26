Reshift
=======

Repeated Pitch-shifting for Frequency Discretization as Artistic Effect for Speech Signals.

The goal is to develop a real-time capable pitch discretization effect named *reshift*  similar to *Autotune* and generate artistic effects by repeatedly applying it to a speech signal.


Directories
-----------

* notebooks: Jupyter Notebooks for developing algorithms and testing

* py: implementation of the developed algorithms

* pd: realtime implementation of Reshift and different audio effects using Reshift


Pitch-Discretization
--------------------

A pitch discretization effect consists of

* a pitch-tracking algorithm

* a nonlinear frequency scale for the target pitch

* and of a pitch-shifting algorithm.

For pitch-tracking, *pYIN* is used and for pitch-shifting, the *Rollers* algorithm has been chosen.


Audio Effects:
--------------

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

* pd/parallelism.pd: template for using many instances of Reshift in separate subprocesses using reshift~_process.pd

* pd/reshift~.pd: the Reshift pitch-discretization effect


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
-----------

* get the anaconda installer for your system from [here](https://www.anaconda.com/products/individual)

* install anaconda

* deactivate conda's base environment with
        conda config --set auto_activate_base false

* install dependencies

```
conda activate base
pip3 install presets librosa snakeviz
sudo apt install gcc make linux-headers-$(uname -r) build-essential automake autoconf libtool gettext git libjack-jackd2-dev libasound2-dev
sudo apt install libaubio-dev
pd/make_pd.sh
pd/make_libs.sh
```

