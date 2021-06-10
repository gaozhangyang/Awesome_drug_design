Literature of Deep Learning for Drug Design
*******************************************

This is a paper list about deep learning for graphs. Some works may be ignored to highlight the really important advances, including

#. low-quality preprint papers
#. papers without open-source python code


.. contents::
    :local:
    :depth: 2

.. sectnum::
    :depth: 2

.. role:: venue(strong)
.. role:: model(emphasis)
.. role:: content(literal)



Generative
===========

`Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules <https://pubs.acs.org/doi/pdf/10.1021/acscentsci.7b00572>`_
    | :venue:`ACS central science 4, no. 2 (2018): 268-276.`
    | `AutoEncoder: SMILES string-->latent vector-->SMILES string`
    | `Convert discrete molecules to a multidimensional continuous representation. Generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds.`


`Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks <https://pubs.acs.org/doi/full/10.1021%2Facscentsci.7b00512>`_
    | :venue:`ACS central science 4, no. 1 (2018): 120-131.`
    | :model:`Stacked LSTMs, SMILE-->LSTM-->SMILE`
    | :content:`Generate molecule SMILEs via stacked LSTMs. Fine tune this model on smaller dataset to get desired molecules, so called transfer learning.`


`Multi-Objective Molecule Generation using Interpretable Substructures <https://arxiv.org/pdf/2002.03244.pdf>`_
    | :venue:`ICML(2020)`
    | :model:`TODO`
    | :content:`Mix molecules with different desired properties to produce the final compound.`
    | `Github(PyTorch) <https://github.com/wengong-jin/multiobj-rationale>`__


`Improving Molecular Design by Stochastic Iterative Target Augmentation <https://arxiv.org/pdf/2002.04720.pdf>`_
    | :venue:`ICML(2020)`
    | :model:`Set2Set and HierGNN`
    | :content:`Augment training data by adding valid generations evaluared by a pre-trained filter.`


`Hierarchical Generation of Molecular Graphs using Structural Motifs <https://arxiv.org/pdf/2002.03230.pdf>`_
    | :venue:`ICML(2020)`
    | :model:`Hierarchical VAE`
    | :content:`TODO`
    | `Github(PyTorch) <https://github.com/wengong-jin/hgraph2graph>`__


`A Generative Model for Molecular Distance Geometry <https://arxiv.org/pdf/1909.11459.pdf>`_
    | :venue:`ICML(2020)`
    | :model:`CVAE`
    | :content:`Generate 3d structure from graph`
    | `Github(Tensorflow) <https://github.com/gncs/graphdg>`__


`Graphaf: a flow-based autoregressive model for molecular graph generation <https://arxiv.org/pdf/2001.09382.pdf>`_
    | :venue:`ICLR(2020)`
    | :model:`TODO`
    | :content:`TODO`
    | `Github(PyTorch) <https://github.com/DeepGraphLearning/GraphAF>`__


Predictive
===========

`Convolutional Networks on Graphs for Learning Molecular Fingerprints <https://arxiv.org/abs/1509.09292>`_
    | :venue:`NIPS 2015`
    | :model:`Prior of GNN`
    | :content:`Learning the graph embedding with message passing.`
    | `Github(PyTorch) <https://github.com/aksub99/molecular-vae>`__
    | `Github(Theano+Keras) <https://github.com/HIPS/molecule-autoencoder>`__


`Molecule Attention Transformer <https://arxiv.org/pdf/2002.08264.pdf>`_
    | :venue:`NeurIPS(workshop) 2019`
    | :model:`transformer`
    | :content:`Argument the attention matrix with distance and ajacency matrix.`
    | `Github(Pytorch) <https://github.com/ardigen/MAT>`__


`Directional Message Passing for Molecular Graphs <https://arxiv.org/pdf/2003.03123.pdf>`__
    | :venue:`ICLR(2020)`
    | :model:`GNN`
    | :content:`TODO`
    | `Github(PyTorch) <https://github.com/klicperajo/dimenet>`__


`ProteinGCN: Protein model quality assessment using graph convolutional networks <https://www.biorxiv.org/content/biorxiv/early/2020/04/07/2020.04.06.028266.full.pdf>`__
    | :venue:`BioRxiv 2020`
    | :model:`GCN`
    | :content:`GCN+Pooling`
    | `Github(Pytorch) <https://github.com/malllabiisc/ProteinGCN>`__


`Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties <https://arxiv.org/pdf/2009.12710.pdf>`_
    | :venue:`ICDM 2020`
    | :model:`Heterogeneous Molecular Graph Neural Networks`
    | :content:`High-order graph convolution, considering interactions between functional groups`
    | `Github(PyTorch) <https://github.com/shuix007/HMGNN>`__


`TrimNet: learning molecular representation from triplet messages for biomedicine <https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbaa266/5955940>`_
    | :venue:`Briefings in Bioinformatics (2020)`
    | :model:`A variant of GAT(Graph Attention model)+Set2Set+Focal loss`
    | :content:`Use graph attention machanism to learn node features, then aggregate them with Set2Set, and finally optimize the model via Focal loss.`
    | `Github(PyTorch) <https://github.com/yvquanli/TrimNet>`__


`Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures <https://arxiv.org/abs/2011.07457>`_
    | :venue:`NIPS 2020`
    | :model:`Hierarchical graph convolution, i.e., local and global message passing.`
    | :content:`Combine local and gloable message passing in the same layer to predict physicochemical properties`
    | `Github(PyTorch) <https://github.com/zetayue/MXMNet>`__



Self-supervised
================
`Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization <https://arxiv.org/pdf/1908.01000.pdf>`_
    | :venue:`ICLR(2020)`
    | :model:`Adversarial self-supervised loss + supervised loss`
    | :content:`Maximize Mutual Information at each layer`
    | `Github(PyTorch) <https://github.com/fanyun-sun/InfoGraph>`__


`Self-Supervised Graph Transformer on Large-Scale Molecular Data <https://drug.ai.tencent.com/publications/GROVER.pdf>`_
    | :venue:`NIPS(2020)`
    | :model:`Transformer+GNN, selfsupervised learning`
    | :content:`Transformer + GNN + node/edge/graph level self-supervised tasks. Downstream tasks include classification and regression.`
    | `Github(PyTorch) <https://github.com/tencent-ailab/grover>`__


Retrosynthesis
==============

`Learning Graph Models for Template-Free Retrosynthesis <https://arxiv.org/pdf/2006.07038.pdf>`_
    | :venue:`ICML workshop(2020)`
    | :model:`Two stages: (1) predict edit (2) add leaving groups`
    | :content:`Use two stage methods to achieve better results.`
    | `Github(PyTorch) <https://github.com/uta-smile/RetroXpert>`__
