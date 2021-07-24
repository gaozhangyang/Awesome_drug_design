.. drug_design documentation master file, created by
   sphinx-quickstart on Mon Jun 14 02:49:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Literature of Deep Learning for Drug Design
*******************************************

`Click for more details <https://gaozhangyang.github.io/Awesome_drug_design/index.html>`__

.. https://gist.github.com/ionelmc/e876b73e2001acd2140f#tables rst教程
.. https://www.jianshu.com/p/1885d5570b37


This is a paper list about deep learning for drug design. Some works may be ignored to highlight the really important advances, including

#. low-quality preprint papers
#. papers without open-source python code

.. .. raw:: html

..     <embed>
..         <style> .red {color:red} </style>
..         <style> .green {color:green} </style>
..     </embed>

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
      | :venue:`ACS central science 4, no. 2 (2018): 268-276.` Gómez-Bombarelli, Rafael, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik.
      | :venue:`Sketch:` AutoEncoder: SMILES string-->latent vector-->SMILES string. Convert discrete molecules to a multidimensional continuous representation. Generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds.


`Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks <https://pubs.acs.org/doi/full/10.1021%2Facscentsci.7b00512>`_
      | :venue:`ACS central science 4, no. 1 (2018): 120-131.` Segler, Marwin HS, Thierry Kogej, Christian Tyrchan, and Mark P. Waller.
      | :venue:`Sketch:` Stacked LSTMs, SMILE-->LSTM-->SMILE. Generate molecule SMILEs via stacked LSTMs. Fine tune this model on smaller dataset to get desired molecules, so called transfer learning.

      
`Multi-Objective Molecule Generation using Interpretable Substructures <https://arxiv.org/pdf/2002.03244.pdf>`_
      | :venue:`ICML(2020)` Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
      | :venue:`Sketch:` Mix molecules with different desired properties to produce the final compound.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/wengong-jin/multiobj-rationale>`__ :red:`Errors.` The code doesn't works, possiblely because the different version of the 'rdkit' package.


`Improving Molecular Design by Stochastic Iterative Target Augmentation <https://arxiv.org/pdf/2002.04720.pdf>`_
      | :venue:`ICML(2020)` Somnath, Vignesh Ram, Charlotte Bunne, Connor W. Coley, Andreas Krause, and Regina Barzilay.`
      | :venue:`Sketch:` Set2Set and HierGNN
      | :venue:`Code:` `Github(PyTorch) <https://github.com/yangkevin2/icml2020-stochastic-iterative-target-augmentation>`__  :red:`Errors.` Due to the unstable updation of chemprop, the code doesn't work.


`Hierarchical Generation of Molecular Graphs using Structural Motifs <https://arxiv.org/pdf/2002.03230.pdf>`_
      | :venue:`ICML(2020)` Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
      | :venue:`Sketch:` Hierarchical VAE
      | :venue:`Code:` `Github(PyTorch) <https://github.com/wengong-jin/hgraph2graph>`__ :red:`Errors.` The code doesn't works, possiblely because the different version of the 'rdkit' package.
      

`A Generative Model for Molecular Distance Geometry <https://arxiv.org/pdf/1909.11459.pdf>`_
      | :venue:`ICML(2020)` Simm, Gregor NC, and José Miguel Hernández-Lobato.
      | :venue:`Sketch:` Generate 3d structure from graph via CVAE
      | :venue:`Code:` `Github(Tensorflow) <https://github.com/gncs/graphdg>`__


`Graphaf: a flow-based autoregressive model for molecular graph generation <https://arxiv.org/pdf/2001.09382.pdf>`_
      | :venue:`ICLR(2020)` Shi, Chence, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. 
      | :venue:`Sketch:` TODO
      | :venue:`Code:` `Github(PyTorch) <https://github.com/DeepGraphLearning/GraphAF>`__ :green:`Good!`

      `A Two-Step Graph Convolutional Decoder for Molecule Generation <https://arxiv.org/pdf/1906.03412.pdf>`_
      | :venue:`arxiv 2019` Bresson, Xavier, and Thomas Laurent.
      | :venue:`Sketch:` Generation: 1.Predicting the number of each types of atoms 2.Link prediction
      | :venue:`Code:` None


`A model to search for synthesizable molecules <https://arxiv.org/pdf/1906.05221.pdf>`_
      | :venue:`NIPS 2019` Bradshaw, John, Brooks Paige, Matt J. Kusner, Marwin HS Segler, and José Miguel Hernández-Lobato. 
      | :venue:`Sketch:` Generation: 1. hidden-->reactants  2. reactants-->products
      | :venue:`Code:` `Github(PyTorch) <https://github.com/john-bradshaw/molecule-chef>`__


`Learning to navigate the synthetically accessible chemical space using reinforcement learning <http://proceedings.mlr.press/v119/gottipati20a/gottipati20a.pdf>`_
      | :venue:`ICML 2020` Gottipati, Sai Krishna, Boris Sattarov, Sufeng Niu, Yashaswi Pathak, Haoran Wei, Shengchao Liu, Simon Blackburn et al.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(Other) <https://github.com/99andBeyond/Apollo1060>`__ :red:`Incomplete!`


`Molecular hypergraph grammar with its application to molecular optimization <https://arxiv.org/pdf/1809.02745.pdf>`_
      | :venue:`ICML 2019` Kajino, Hiroshi.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(Pytorch) <https://github.com/ibm-research-tokyo/graph_grammar>`__


Predictive
===========

`Convolutional Networks on Graphs for Learning Molecular Fingerprints <https://arxiv.org/abs/1509.09292>`_
      | :venue:`NIPS 2015` Duvenaud, David, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P. Adams.
      | :venue:`Sketch:` Learning the graph embedding with message passing.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/aksub99/molecular-vae>`__  `Github(Theano+Keras) <https://github.com/HIPS/molecule-autoencoder>`__


`Molecule Attention Transformer <https://arxiv.org/pdf/2002.08264.pdf>`_
      | :venue:`NeurIPS(workshop) 2019` Maziarka, Łukasz, Tomasz Danel, Sławomir Mucha, Krzysztof Rataj, Jacek Tabor, and Stanisław Jastrzębski.
      | :venue:`Sketch:` `Argument the attention matrix with distance and ajacency matrix.`
      | :venue:`Code:` `Github(Pytorch) <https://github.com/ardigen/MAT>`__ :green:`Good!`


`Directional Message Passing for Molecular Graphs <https://arxiv.org/pdf/2003.03123.pdf>`__
      | :venue:`ICLR(2020)` Klicpera, Johannes, Janek Groß, and Stephan Günnemann.
      | :venue:`Sketch:` TODO
      | :venue:`Code:` `Github(Tensorflow) <https://github.com/klicperajo/dimenet>`__


`ProteinGCN: Protein model quality assessment using graph convolutional networks <https://www.biorxiv.org/content/biorxiv/early/2020/04/07/2020.04.06.028266.full.pdf>`__
      | :venue:`BioRxiv 2020` Sanyal, Soumya, Ivan Anishchenko, Anirudh Dagar, David Baker, and Partha Talukdar.
      | :venue:`Sketch:` GCN+Pooling
      | :venue:`Code:` `Github(Pytorch) <https://github.com/malllabiisc/ProteinGCN>`__


`Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties <https://arxiv.org/pdf/2009.12710.pdf>`_
      | :venue:`ICDM 2020` Shui, Zeren, and George Karypis.
      | :venue:`Sketch:` Heterogeneous Molecular Graph Neural Networks. High-order graph convolution, considering interactions between functional groups
      | :venue:`Code:` `Github(PyTorch) <https://github.com/shuix007/HMGNN>`__


`TrimNet: learning molecular representation from triplet messages for biomedicine <https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbaa266/5955940>`_
      | :venue:`Briefings in Bioinformatics (2020)` Li, Pengyong, Yuquan Li, Chang-Yu Hsieh, Shengyu Zhang, Xianggen Liu, Huanxiang Liu, Sen Song, and Xiaojun Yao.
      | :venue:`Sketch:` A variant of GAT(Graph Attention model)+Set2Set+Focal loss. Use graph attention machanism to learn node features, then aggregate them with Set2Set, and finally optimize the model via Focal loss.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/yvquanli/TrimNet>`__


`Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures <https://arxiv.org/abs/2011.07457>`_
      | :venue:`NIPS 2020` Zhang, Shuo, Yang Liu, and Lei Xie.
      | :venue:`Sketch:` Hierarchical graph convolution, i.e., local and global message passing.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/zetayue/MXMNet>`__ :green:`Good!`


`Directional message passing for molecular graphs <https://arxiv.org/pdf/2003.03123.pdf>`_
      | :venue:`ICLR 2020` Klicpera, Johannes, Janek Groß, and Stephan Günnemann.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(Tensorflow) <https://github.com/klicperajo/dimenet>`__ 

`Molecular property prediction: A multilevel quantum interactions modeling perspective <https://arxiv.org/pdf/1906.11081.pdf>`_
      | :venue:`AAAI 2019` Lu, Chengqiang, Qi Liu, Chao Wang, Zhenya Huang, Peize Lin, and Lixin He
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(PyTorch) <https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/mgcn_predictor.py>`__ 


Self-supervised
================
`Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization <https://arxiv.org/pdf/1908.01000.pdf>`_
      | :venue:`ICLR(2020)` Sun, Fan-Yun, Jordan Hoffmann, Vikas Verma, and Jian Tang.
      | :venue:`Sketch:` Adversarial self-supervised loss + supervised loss. aximize Mutual Information at each layer.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/fanyun-sun/InfoGraph>`__


`Self-Supervised Graph Transformer on Large-Scale Molecular Data <https://drug.ai.tencent.com/publications/GROVER.pdf>`_
      | :venue:`NIPS(2020)` Rong, Yu, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang.
      | :venue:`Sketch:` Transformer + GNN + node/edge/graph level self-supervised tasks. Downstream tasks include classification and regression.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/tencent-ailab/grover>`__


Retrosynthesis & Molecular Optimization
=======================================
`Retrosynthetic Reaction Prediction Using Neural Sequence-to-Sequence Models <https://arxiv.org/ftp/arxiv/papers/1706/1706.01643.pdf>`_
      | :venue:`ACS central science 3, no. 10 (2017): 1103-1113.` Liu, Bowen, Bharath Ramsundar, Prasad Kawthekar, Jade Shi, Joseph Gomes, Quang Luu Nguyen, Stephen Ho, Jack Sloane, Paul Wender, and Vijay Pande.
      | :venue:`Sketch:` SMILES-->seq2seq-->SMILES
      | :venue:`Code` `Github(Tensorflow) <https://github.com/pandegroup/reaction_prediction_seq2seq.git>`__


`Predicting retrosynthetic reactions using self-corrected transformer neural networks <https://arxiv.org/ftp/arxiv/papers/1907/1907.01356.pdf>`_
      | :venue:`Journal of chemical information and modeling 60, no. 1 (2019): 47-55.` Zheng, Shuangjia, Jiahua Rao, Zhongyue Zhang, Jun Xu, and Yuedong Yang.
      | :venue:`Sketch` Building a Transformer-based syntax corrector to automatically correct the syntax of unreasonable SMILES strings for improving the performances.
      | :venue:`Code` `Github(PyTorch) <https://github.com/sysu-yanglab/Self-Corrected-Retrosynthetic-Reaction-Predictor>`__


`Automatic Retrosynthetic Route Planning Using Template-Free Models <https://arxiv.org/ftp/arxiv/papers/1906/1906.02308.pdf>`_
      | :venue:`Chemical Science, 11(12):3355–3364, 2020.` Lin, Kangjie, Youjun Xu, Jianfeng Pei, and Luhua Lai.
      | :venue:`Sketch:` Using Transformer for one-step retrosynthesis. Then, combining the Monte Carlo Tree Search for multi-step retrosynthesis.
      | :venue:`Code:` `Github(Tensorflow) <https://github.com/connorcoley/retrotemp>`__


`RetroXpert: Decompose Retrosynthesis Prediction Like A Chemist <https://arxiv.org/pdf/2011.02893.pdf>`_
      | :venue:`NIPS(2020)` Yan, Chaochao, Qianggang Ding, Peilin Zhao, Shuangjia Zheng, Jinyu Yang, Yang Yu, and Junzhou Huang.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(PyTorch) <https://github.com/uta-smile/RetroXpert>`__  :green:`Minor errors.` The code works with a few tweaks.


`Learning Graph Models for Template-Free Retrosynthesis <https://arxiv.org/pdf/2006.07038.pdf>`_
      | :venue:`ICML workshop(2020)` `Somnath, Vignesh Ram, Charlotte Bunne, Connor W. Coley, Andreas Krause, and Regina Barzilay.`
      | :venue:`Sketch:` Use two stage methods to achieve better results: (1) predict edit (2) add leaving groups


`RetroPrime: A Diverse, Plausible and Transformer-based Method for Single-Step Retrosynthesis Predictions <https://chemrxiv.org/articles/preprint/RetroPrime_A_Chemistry-Inspired_and_Transformer-based_Method_for_Retrosynthesis_Predictions/12971942>`_
      | :venue:`Chemical Engineering Journal 420 (2021): 129845.` Wang, Xiaorui, Yuquan Li, Jiezhong Qiu, Guangyong Chen, Huanxiang Liu, Benben Liao, Chang-Yu Hsieh, and Xiaojun Yao.
      | :venue:`Sketch:`
      | :venue:`Code:` `Github(PyTorch) <https://github.com/wangxr0526/RetroPrime>`__ 


`A graph to graphs framework for retrosynthesis prediction <http://proceedings.mlr.press/v119/shi20d/shi20d.pdf>`_
      | :venue:`PMLR (2020)` Shi, Chence, Minkai Xu, Hongyu Guo, Ming Zhang, and Jian Tang.
      | :venue:`Sketch:`
      | :venue:`Code:`  None 


`Molecule Optimization via Fragment-based Generative Models <https://arxiv.org/pdf/2012.04231.pdf>`_
      | :venue:`arxiv 2021` Chen, Ziqi, Martin Renqiang Min, Srinivasan Parthasarathy, and Xia Ning.
      | :venue:`Sketch:` Using molecular graph and junction tree to learn embeddings. Then, 1. predicting site 2. remove fragment 3. predict connections and fragments.
      | :venue:`Code:` `Github(PyTorch) <https://github.com/ziqi92/Modof>`__


`Retrosynthesis Prediction with Conditional Graph Logic Network <https://arxiv.org/pdf/2001.01408.pdf>`_
      | :venue:`NIPS 2019` Dai, Hanjun, Chengtao Li, Connor W. Coley, Bo Dai, and Le Song.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(PyTorch) <https://github.com/Hanjun-Dai/GLN>`__

`Retrognn: Approximating retrosynthesis by graph neural networks for de novo drug design <https://arxiv.org/ftp/arxiv/papers/2011/2011.13042.pdf>`_
      | :venue:`NIPS(workshop) 2020` Liu, Cheng-Hao, Maksym Korablyov, Stanisław Jastrzębski, Paweł Włodarczyk-Pruszyński, Yoshua Bengio, and Marwin HS Segler. "
      | :venue:`Sketch:` 
      | :venue:`Code:` None

Association
=====================

`Discovering Protein Drug Targets Using Knowledge Graph Embeddings <https://aran.library.nuigalway.ie/bitstream/handle/10379/15375/main_dti.pdf?sequence=1&isAllowed=n>`_
      | :venue:`Bioinformatics (2020)` Mohamed, Sameh K., Vít Nováček, and Aayah Nounu.
      | :venue:`Asign a learnable embedding/parameter for each node and optimizing these embedding via true/false facts on the Knowledge graph.`
      | :venue:`Code:`  `Numpy(lack of training code) <http://drugtargets.insight-centre.org/download.html>`__

`BNPMDA: Bipartite Network Projection for MiRNA–Disease Association prediction <https://watermark.silverchair.com/bty333.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAt4wggLaBgkqhkiG9w0BBwagggLLMIICxwIBADCCAsAGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMpRlzvbNSEjBqQlcsAgEQgIICkYeklyup8JUnfRkAFMgB7jWi7zVkHvFRCa0OoMnW4XH4_VfwsoZyRS6P9_2ftDBIBcyeLeQib9ynPV1w7gfx62mviGRFe7zH7J_e2sjIyJODZAubdy4LuDrtQf_LRybWriIGX-7ASyqvSEaE9tB2GfOWkRgHsHjB8T7srP-ZvRZjk38m6ftmwn3R3x4_36ACST4E7KZ1bPfrLmpKU_EeWfMNuOZ7SJmyLJpiiRGZTYQ6ymfaAjNAYlPtSNS5MrM5XeYTWnZqIbKiTaBMvHts6IwCb__26asfc3gh9GfhIZ8rIWGPV0EmCcob5S4ToXUUd_BKPB8GoWIgyKM-uhGK6rKakNm7m3HzM1lULSHbqT_1kE00vFZGI2KhDYOTw5YeBCiJxEZkkLvg1brlhvyozFws9ZqtY3X1Oel_ki92r7FWYc98hM_4z15mAHD5w22HjJoUoqtqPKN_jVAoqvzpCwqNBMudmy_mjGNxRCbO0B-g5UDPMDix4gz_Bm712TROX_OF_z1ipTG-6RvO6l9RqVzZfmeKJotIKXRprMjZ_EkfHhPQCTgovbIlY9RPUESqamCoUVAuhKGsicWk7LBgIoNW1KVYad3weFO8YqeLeXF2RLQqG3KA_lrSwG4Nl0bC4mxGDKoWcc15jBSNPJynpkXJRSbgmzrlCwZbHEc_UnITsYH2DQpOZFGmJYNBum8Xueo3kmGhXBh76Z0Y96AqTIHMTmbDlu6GxxblMUg2zAjCuLnOyBf4buzSdc5ZfGGaIIEttgorXjBjIqK-tRUWtIuhYgWwuf9CAZgrv8YWv0UK45zqzNQtfauQ_CWBVOBRtzV21y8gOA8qoVPWEyuAmW-MARNBkxYj3qEWMyrXF4w1qg>`_
      | :venue:`Bioinformatics (2018)` Chen, Xing, Di Xie, Lei Wang, Qi Zhao, Zhu-Hong You, and Hongsheng Liu.
      | :venue:`Bipartite Network Projection for MiRNA-Disease Association prediction based on the known miRNA-disease association, intergrated miRNA similarity and integrated disease similarity.`
      | :venue:`Code`   Can't access.

`Network analyses identify liver-specific targets for treating liver diseases <https://www.embopress.org/doi/pdf/10.15252/msb.20177703>`_
      | :venue:`Molecular systems biology (2017)` Lee, Sunjae, Cheng Zhang, Zhengtao Liu, Martina Klevstig, Bani Mukhopadhyay, Mattias Bergentall, Resat Cinar et al.
      | :venue:`We performed integrative network analyses to identify targets that can be used for effectively treating liver diseases with minimal side effects`
      | :venue:`Code`   None

`Human disease MiRNA inference by combining target information based on heterogeneous manifolds <https://reader.elsevier.com/reader/sd/pii/S1532046418300327?token=2799B1B133D1CDCB9910B0884EEFF5D14FD849145E387A065E566D3578C936B52FF335AE24F18EA9EB60C9D56C104DC4&originRegion=us-east-1&originCreation=20210719042558>`_
      | :venue:`Journal of biomedical informatics  (2018)` Ding, Pingjian, Jiawei Luo, Cheng Liang, Qiu Xiao, and Buwen Cao.
      | :venue:`we developed a novel algorithm, named inference of Disease-related MiRNAs based on Heterogeneous Manifold (DMHM), to accurately and efficiently identify miRNA-disease associations by integrating multi-omics data`
      | :venue:`Code`   Can't access.


`Caster: Predicting drug interactions with chemical substructure representation <https://arxiv.org/pdf/1911.06446.pdf>`_
      | :venue:`AAAI 2020` Huang, Kexin, Cao Xiao, Trong Hoang, Lucas Glass, and Jimeng Sun.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(PyTorch) <https://github.com/kexinhuang12345/CASTER/tree/master/DDE>`__


`Drug-drug interaction prediction based on knowledge graph embeddings and convolutional-LSTM network <https://arxiv.org/pdf/1908.01288.pdf>`_
      | :venue:`the 10th ACM international conference on bioinformatics` Karim, Md Rezaul, Michael Cochez, Joao Bosco Jares, Mamtaz Uddin, Oya Beyan, and Stefan Decker.
      | :venue:`Sketch:` 
      | :venue:`Code:` `Github(Tensorflow&Keras) <https://github.com/rezacsedu/Drug-Drug-Interaction-Prediction>`__
