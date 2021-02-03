Reshift
=======

Repeated Pitch-shifting for Frequency Discretization as Artistic Effect for Speech Signals.

The goal is to develop a real-time capable pitch discretization effect similar to *Autotune* and generate an artistic effect named *reshift* by repeatedly applying it to a speech signal.


Directories
-----------

* notebooks: Jupyter Notebooks for developing algorithms and testing

* py: implementation of the developed algorithms


Pitch-Discretization
--------------------

A pitch discretization effect consists of

* a pitch-tracking algorithm

* a nonlinear frequency scale for the target pitch

* and of a pitch-shifting algorithm.

For pitch-tracking, *pYIN* should be used and for pitch-shifting, the *Rollers* algorithm has been chosen.


Notebooks
---------

These Jupyter Notebooks document the process of designing the *reshift* algorithm.
The main task is the development of the pitch-discretization effect using *pYIN* and *Rollers*.

As a first step, [pitch discretization with librosa](notebooks/Pitch-discretization with librosa.ipynb) has been implemented, using librosa's pitch-shifting algorithm.
Librosa does not support a pitch-shifting algorithm for a time variant pitch shifting factor.

So as a next step, a [time variant pitch-shifting algorithm using overlap and add](notebooks/OLA Pitch Shifting.ipynb) has been developed.
This serves as a simple implementation of a pitch-shifting algorithm to do pitch-discretization, as a comparison to the final used *Rollers* pitch-shifting algorithm.

The development and parameter adjustment of the OLA pitch-discretization effect is done in [Algorithms and Parameters](notebooks/Algorithms and Parameters.ipynb).
This notebook uses the developed algorithms in the **py/** directory.



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
pip3 install presets librosa
```

