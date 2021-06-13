Literature of Deep Learning for Drug Design
*******************************************

.. https://gist.github.com/ionelmc/e876b73e2001acd2140f#tables rst教程
.. https://www.jianshu.com/p/1885d5570b37


This is a paper list about deep learning for graphs. Some works may be ignored to highlight the really important advances, including

#. low-quality preprint papers
#. papers without open-source python code



.. raw:: html

    <embed>
        <style> .red {color:red} </style>
        <style> .green {color:green} </style>
    </embed>

.. role:: red
.. role:: green


.. contents::
    :local:
    :depth: 2

.. sectnum::
    :depth: 2

.. role:: venue(strong)
.. role:: model(emphasis)
.. role:: content(literal)



Dataset
========
`Pubchem <https://pubchem.ncbi.nlm.nih.gov/classification/#hid=1>`_

`Therapeutics Data Commons <https://tdcommons.ai/>`_

`MoleculeNet <http://moleculenet.ai/>`_


Generative
===========

`Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules <https://pubs.acs.org/doi/pdf/10.1021/acscentsci.7b00572>`_
    :Info: :venue:`ACS central science 4, no. 2 (2018): 268-276.` Gómez-Bombarelli, Rafael, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik.
    :Sketch: AutoEncoder: SMILES string-->latent vector-->SMILES string. Convert discrete molecules to a multidimensional continuous representation. Generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds.


`Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks <https://pubs.acs.org/doi/full/10.1021%2Facscentsci.7b00512>`_
    :Info: :venue:`ACS central science 4, no. 1 (2018): 120-131.` Segler, Marwin HS, Thierry Kogej, Christian Tyrchan, and Mark P. Waller.
    :Sketch: Stacked LSTMs, SMILE-->LSTM-->SMILE. Generate molecule SMILEs via stacked LSTMs. Fine tune this model on smaller dataset to get desired molecules, so called transfer learning.

    
`Multi-Objective Molecule Generation using Interpretable Substructures <https://arxiv.org/pdf/2002.03244.pdf>`_
    :Info: :venue:`ICML(2020)` Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
    :Sketch: Mix molecules with different desired properties to produce the final compound.
    :Code: `Github(PyTorch) <https://github.com/wengong-jin/multiobj-rationale>`__ :red:`Errors.` The code doesn't works, possiblely because the different version of the 'rdkit' package.


`Improving Molecular Design by Stochastic Iterative Target Augmentation <https://arxiv.org/pdf/2002.04720.pdf>`_
    :Info: :venue:`ICML(2020)` Somnath, Vignesh Ram, Charlotte Bunne, Connor W. Coley, Andreas Krause, and Regina Barzilay.`
    :Sketch: Set2Set and HierGNN
    :Code: `Github(PyTorch) <https://github.com/yangkevin2/icml2020-stochastic-iterative-target-augmentation>`__  :red:`Errors.` Due to the unstable updation of chemprop, the code doesn't work.


`Hierarchical Generation of Molecular Graphs using Structural Motifs <https://arxiv.org/pdf/2002.03230.pdf>`_
    :Info: :venue:`ICML(2020)` Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
    :Sketch: Hierarchical VAE
    :Code: `Github(PyTorch) <https://github.com/wengong-jin/hgraph2graph>`__ :red:`Errors.` The code doesn't works, possiblely because the different version of the 'rdkit' package.
    

`A Generative Model for Molecular Distance Geometry <https://arxiv.org/pdf/1909.11459.pdf>`_
    :Info: :venue:`ICML(2020)` Simm, Gregor NC, and José Miguel Hernández-Lobato.
    :Sketch: Generate 3d structure from graph via CVAE
    :Code: `Github(Tensorflow) <https://github.com/gncs/graphdg>`__


`Graphaf: a flow-based autoregressive model for molecular graph generation <https://arxiv.org/pdf/2001.09382.pdf>`_
    :Info: :venue:`ICLR(2020)` Shi, Chence, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. 
    :Sketch: TODO
    :Code: `Github(PyTorch) <https://github.com/DeepGraphLearning/GraphAF>`__ :green:`Good!`


Predictive
===========

`Convolutional Networks on Graphs for Learning Molecular Fingerprints <https://arxiv.org/abs/1509.09292>`_
    :Info: :venue:`NIPS 2015` Duvenaud, David, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P. Adams.
    :Sketch: Learning the graph embedding with message passing.
    :Code: `Github(PyTorch) <https://github.com/aksub99/molecular-vae>`__  `Github(Theano+Keras) <https://github.com/HIPS/molecule-autoencoder>`__


`Molecule Attention Transformer <https://arxiv.org/pdf/2002.08264.pdf>`_
    :Info: :venue:`NeurIPS(workshop) 2019` Maziarka, Łukasz, Tomasz Danel, Sławomir Mucha, Krzysztof Rataj, Jacek Tabor, and Stanisław Jastrzębski.
    :Sketch: `Argument the attention matrix with distance and ajacency matrix.`
    :Code: `Github(Pytorch) <https://github.com/ardigen/MAT>`__ :green:`Good!`


`Directional Message Passing for Molecular Graphs <https://arxiv.org/pdf/2003.03123.pdf>`__
    :Info: :venue:`ICLR(2020)` Klicpera, Johannes, Janek Groß, and Stephan Günnemann.
    :Sketch: TODO
    :Code: `Github(Tensorflow) <https://github.com/klicperajo/dimenet>`__


`ProteinGCN: Protein model quality assessment using graph convolutional networks <https://www.biorxiv.org/content/biorxiv/early/2020/04/07/2020.04.06.028266.full.pdf>`__
    :Info: :venue:`BioRxiv 2020` Sanyal, Soumya, Ivan Anishchenko, Anirudh Dagar, David Baker, and Partha Talukdar.
    :Sketch: GCN+Pooling
    :Code: `Github(Pytorch) <https://github.com/malllabiisc/ProteinGCN>`__


`Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties <https://arxiv.org/pdf/2009.12710.pdf>`_
    :Info: :venue:`ICDM 2020` Shui, Zeren, and George Karypis.
    :Sketch: Heterogeneous Molecular Graph Neural Networks. High-order graph convolution, considering interactions between functional groups
    :Code: `Github(PyTorch) <https://github.com/shuix007/HMGNN>`__


`TrimNet: learning molecular representation from triplet messages for biomedicine <https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbaa266/5955940>`_
    :Info: :venue:`Briefings in Bioinformatics (2020)` Li, Pengyong, Yuquan Li, Chang-Yu Hsieh, Shengyu Zhang, Xianggen Liu, Huanxiang Liu, Sen Song, and Xiaojun Yao.
    :Sketch: A variant of GAT(Graph Attention model)+Set2Set+Focal loss. Use graph attention machanism to learn node features, then aggregate them with Set2Set, and finally optimize the model via Focal loss.
    :Code: `Github(PyTorch) <https://github.com/yvquanli/TrimNet>`__


`Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures <https://arxiv.org/abs/2011.07457>`_
    :Info: :venue:`NIPS 2020` Zhang, Shuo, Yang Liu, and Lei Xie.
    :Sketch: Hierarchical graph convolution, i.e., local and global message passing.
    :Code: `Github(PyTorch) <https://github.com/zetayue/MXMNet>`__ :green:`Good!`



Self-supervised
================
`Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization <https://arxiv.org/pdf/1908.01000.pdf>`_
    :Info: :venue:`ICLR(2020)` Sun, Fan-Yun, Jordan Hoffmann, Vikas Verma, and Jian Tang.
    :Sketch: Adversarial self-supervised loss + supervised loss. aximize Mutual Information at each layer.
    :Code: `Github(PyTorch) <https://github.com/fanyun-sun/InfoGraph>`__


`Self-Supervised Graph Transformer on Large-Scale Molecular Data <https://drug.ai.tencent.com/publications/GROVER.pdf>`_
    :Info: :venue:`NIPS(2020)` Rong, Yu, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang.
    :Sketch: Transformer + GNN + node/edge/graph level self-supervised tasks. Downstream tasks include classification and regression.
    :Code: `Github(PyTorch) <https://github.com/tencent-ailab/grover>`__


Retrosynthesis
==============

`Learning Graph Models for Template-Free Retrosynthesis <https://arxiv.org/pdf/2006.07038.pdf>`_
    :Info: :venue:`ICML workshop(2020)` `Somnath, Vignesh Ram, Charlotte Bunne, Connor W. Coley, Andreas Krause, and Regina Barzilay.`
    :Sketch: Use two stage methods to achieve better results: (1) predict edit (2) add leaving groups
    :Code: `Github(PyTorch) <https://github.com/uta-smile/RetroXpert>`__  :green:`Minor errors.` The code works with a few tweaks.
