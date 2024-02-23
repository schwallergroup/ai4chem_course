## AI4Chemistry course

The Artificial Intelligence (AI) for Chemistry course will be taught in Spring 2023. It is a course with a lot of hands-on exercises. Experience in Python programming and machine learning (ML) will help you to get up to speed quickly, but we will try to make it as accessible as possible. 

We will make use of [Google Colab](https://colab.research.google.com) to run the code directly in your browser, as there is zero configuration required and we will have access to GPUs free of charge.

A lot of the examples and ideas in this course are taken from the open-source community, which we will properly reference.

## Contributors

This course is being created by the [LIAC team](https://schwallergroup.github.io/team.html). 
Many thanks to all the TAs:

- [Victor Sabanza Gil](https://twitter.com/VictorSabanza)
- [Bojana Ranković](https://twitter.com/6ojaHa)
- [Junwu Chen](https://twitter.com/JunwuChen25)
- [Andres CM Bran](https://twitter.com/drecmb)
- [Jeff Guo](https://twitter.com/JeffGuo__)

## Tentative content

- Python Crash Course & essential libraries (matplotlib, numpy, pandas)
- Cheminformatics toolkits (rdkit)
- Introduction into data science
    - Supervised machine learning (regression, classification)
    - Unsupervised machine learning
    - Data and standardisation
- Deep Learning for Chemistry
    - Property prediction models
    - Inverse Design [@sanchez2018inverse]
    - Reaction prediction and retrosynthesis [@schwaller2022machine]
- Advanced topics in AI for Chemistry
    - Bayesian optimisation for chemical reactions

## Exercises

| Week | Topic | Link to Colab |
|--|--|--|
| 1 | Python and Jupyter | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01a_python_crash_course.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Pandas | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01b_python_essentials_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|
| | Plotting  data | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01c_python_essentials_plotting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 
|  | Intro to RDKit | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01d_rdkit_basics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 
|2 | Supervised ML |<a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/02%20-%20Supervised%20Learning/training_and_evaluating_ml_models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 3 | Introduction to Deep Learning | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/01_intro_to_dl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Graph Neural Network | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/02_graph_nns.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | GNN example - chemprop | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/03_gnn_simple_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 4 | Dimensionality reduction | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/04%20-%20Unsupervised%20Learning/DimensionalityReduction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Clustering | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/04%20-%20Unsupervised%20Learning/Clustering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Pd dimers discovery by kMeans | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/04%20-%20Unsupervised%20Learning/palladium_dimers_discovery.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 5 | De novo molecule generation (VAE) | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/generative_models/notebooks/05%20-%20Generative%20Models/Molecular%20Generative%20Models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 6 | De novo molecule generation (SMILES-LSTM) | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/smiles_lstm/notebooks/06%20-%20Generative%20Models%202/SMILES-LSTM-Walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 7 | Chemical reactions prediction: Template-free methods | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/07%20-%20Reaction%20Prediction/template_free.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 8 | Retrosynthesis: Template-based methods | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/08%20-%20Retrosynthesis/template_based.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 9 | Atom mapping | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/09%20-%20Reaction%20properties/01_atom_mapping.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Reaction fingerprints | <a href="https://colab.research.google.com/github/schwallergroup/drfp/blob/main/notebooks/01_fingerprinting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Yield prediction | <a href="https://colab.research.google.com/github/schwallergroup/drfp/blob/main/notebooks/02_model_explainability.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 10 | Bayesian Optimisation | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/10%20-%20Bayesian%20optimization/Buchwald-Hartwig-BO.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 11 | Model deployment: Git(hub) | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/11%20-%20Model%20deployment/git(hub)_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| | Model deployment: Streamlit | <a href="https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/11%20-%20Model%20deployment/streamlit_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 12 | Guest lecture: AlphaFold2 | <a href="https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


The solutions can be found in this [GitHub repo](https://github.com/schwallergroup/ai4chem_course). Don't forget to leave a star, if you find it useful. 

## Inspiration

The cheminformatics and ML for chemistry have a lively open source community. Here is a collection of inspirational blogs and webpages, from which we discuss examples:

### Cheminformatics / ML for Chemistry
- [Andrew White's Deep Learning for Molecules & Materials book](https://dmol.pub)
- [MolSSI Education Resources](http://education.molssi.org/resources.html#programming)
- [Greg Landrum's RDKit blog](https://greglandrum.github.io/rdkit-blog/)
- [Esben Bjerrum's Cheminformania](https://www.cheminformania.com)
- [iwatobipens' blog](https://iwatobipen.wordpress.com)
- [Rocío Mercado's dl-chem-101](https://github.com/rociomer/dl-chem-101)
- [Jan H. Jensen's Machine Learning Basics](https://sites.google.com/view/ml-basics/home)
- [Pat Walter's Practical Cheminformatics With Open Source Software](https://github.com/PatWalters/practical_cheminformatics_tutorials)
- [Kjell Jorner's blog](https://www.valencekjell.com)

### AI for Science
- [Lewis Tunstall's Deep Learning for Particle Physicists](https://lewtun.github.io/dl4phys/intro.html)
- [Summer school on Statistical Physics & Machine learning](https://leshouches2022.github.io) organised by Florent Krzakala and Lenka Zdeborova, EPFL

### ML & Data Science
- [Practical Deep Learning](https://course.fast.ai)
- [MIT's Intro to Deep Learning](http://introtodeeplearning.com)
- [Aurelien Geron's Hands-on Machine Learning](https://github.com/ageron/handson-ml2)
- [Lewis Tunstall's Introduction to Data Science](https://lewtun.github.io/dslectures/)
- [Natural Language Processing with Transformers by HuggingFace](https://github.com/nlp-with-transformers/notebooks)

Check them out and don't forget to leave a star on GitHub and follow the authors on Twitter, if you like the content. 
Those blogs and webpages have all helped me during the creation of this course (and also before, when I was learning about ML for Chemistry).

## Tweets

{{< tweet pschwllr 1629098793399472130 >}}
