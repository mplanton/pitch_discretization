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

* pd/FX2.pd: audio effects using two instances of Reshift

* pd/FX_many.pd: audio effects using many instances of Reshift

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

