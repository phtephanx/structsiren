Structural Siren
================

.. figure:: images/test_set.png
   :width: 600
   :alt: SSAE on test set after 60 epochs of training
   
   SSAE on test set after 60 epochs of training

Introduction
------------
Structural Siren Autoencoders (`SSAE`) strive to reveal factors of
variation in generative processes by structural disentanglement. 
They are based on Structural Autoencoders (`SAE`), introduced by [LEB21]_, but 
replace their upsampling decoder a with Siren network, proposed by [SIT19]_.
Unlike VAEs, which achieve regularity on the latent space by enforcing
prior distributions on its latent codes,
`SSAEs` embed a hierarchical structural causal model (SCM) into their decoder:

.. math::

    S_i := f_i(PA_i, U_i), (i=1, ..., n)

[LEB21]_ allude to the fact that in an SCM the endogenous variables
:math:`S_i` are not statistically independent but hierarchically dependent.
Instead, the set of noises :math:`U_i`, which are represented by the latent
codes :math:`q` of the `SSAE`, is assumed to be jointly independent. To emulate this
structure, the individual
latent code vectors :math:`q_k` are injected one after another into layers of the decoder.



.. figure:: images/decoder.png
   :width: 600
   :alt: structural decoder
   
Structural Causal Model
-----------------------
The SCM of `SSAE` has the following form:

.. math::

    h_1 = alpha_1 * sin(w_1 * h_0 + b_1) + beta_1
    h_2 = alpha_2 * sin(w_2 * h_1 + b_2) + beta_2
    ...
    h_K = alpha_K * sin(w_K * h_{K-1} + b_K) + beta_K


where :math:`alpha_k` and :math:`beta_k` originating from latent code
:math:`q_k` are subsequently injected into the decoder and modulate its
activations. The advantage of using sinusoidal nonlinearities is to preserve second
and higher-order derivatives during reconstruction, compared to e.g. ReLU. 
It is worthwhile noting that :math:`alpha_k` and :math:`beta_k` 
do not control :math:`h_k` pixelwise but channelwise. [MEH21]_ apply the same
modulation without splitting up the latent vector into the individual codes and 
injecting them subsequently according to causal ordering.


Experiment
----------

* data: 3dshapes_
* 70-10-20 train-dev-test split
* encoder: pre-trained `EfficientNet`_ "b0"
* decoder: 6-layer Siren with 6 :math:`q_k element R??`
* epochs: 60

.. _3dshapes: https://github.com/deepmind/3d-shapes
.. _EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch


Reconstruction & Disentanglement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------+--------+
|       | SSAE12 |
+-------+--------+
| FID   | 12.099 |
+-------+--------+
| DCI-d | 0.687  |
+-------+--------+
| IRS   | 0.677  |
+-------+--------+
| DCI-c | 0.543  |
+-------+--------+

Visual Probing
~~~~~~~~~~~~~~

The latent codes of the test set are visually probed for their disentanglement
by coloring them dependent on the configuration of the ground truth factors
of variation. For instance, code vector :math:`q_1` modulates the color of the wall
and code vector :math:`q_2` the color of the floor.

.. image:: images/codes-to-factors.png
   :width: 800
   :alt: latent codes of test set colored with ground truth configuration


Installation
------------

To install `structsiren`, run:

.. code-block:: python

    pip install -r requirements.txt

Scripts
-------

+ `3dshapes_prepare_data.py`: prepare `3d-shapes` data
+ `3dshapes_train.py`: train Structural Siren with pre-trained `EfficientNet`
  encoder
+ `3dshapes_plot_shapes.py`: plot reconstructions for test data with
  pre-trained model
+  `3dshapes_collect_factors.py`: collect codes for train-dev-test data with
   pre-trained model
+ `3dshapes_measure_disentanglement.py`
+ `3dshapes_codes_to_factors.py`: create scatter plots of codes and colorize
  with manifestations of different ground truth factors
  
References
----------

.. [SIT19] V\. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, en G. Wetzstein, ???Implicit Neural Representations with Periodic Activation Functions???, in arXiv, 2020.
.. [LEB21] F\. Leeb, G. Lanzillotta, Y. Annadani, M. Besserve, S. Bauer, en B. Sch??lkopf, ???Structure by Architecture: Disentangled Representations without Regularization???, arXiv [cs.LG]. 2021.
.. [MEH21] I\. Mehta, M. Gharbi, C. Barnes, E. Shechtman, R. Ramamoorthi, en M. Chandraker, ???Modulated Periodic Activations for Generalizable Local Functional Representations???, arXiv [cs.CV]. 2021.
