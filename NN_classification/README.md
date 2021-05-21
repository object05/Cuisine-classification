# Cuisine-Prediction

### Introduction
- In this notebook, we implement a prediction system that uses [recipe-ingredient data from this Kaggle challenge](https://www.kaggle.com/c/whats-cooking) to predict the cuisine of a given recipe
- We implement and evaluate techniques from the following domains:
    1. **Simple (baseline) heuristics**
        1. Baseline #1: For each ingredient belonging to the given test ingredient list, find the cuisine in which this ingredient is used the most. Among all such cuisines, pick the most frequently occurring cuisine
        2. Baseline #2: Rank the set of training recipes based on number of ingredients common with test ingredient list. Assign weights to recipes based on their rank and add to scores of their corresponding cuisines. Finally, choose the cuisine with the highest score
    2. **Machine Learning**
        1. Neural Network
        2. Support Vector Machine (SVM)
    3. **Machine Learning + Network**
        1. [node2vec](https://github.com/aditya-grover/node2vec) embeddings of the unipartite projection network (where each node is an ingredient) of the recipe-ingredient bipartitite network are first obtained
        1. Neural Network, SVM, GRU are then trained with these embeddings as features 
    4. **Network-based (clustering) heuristics**
        1. Do clustering of ingredients using K-means on [node2vec](https://github.com/aditya-grover/node2vec) embeddings of ingredient-ingredient network (we set K = # cuisines). We analyze if these generated clusters have one-to-one correlation with different cuisines. If yes, we can use these clusters to create a prediction heuristic in which the cuisine having most number of test ingredients (i.e. the cluster having most number of nodes out of a given set of nodes) is picked

### Installation
- Download recipe-ingredient data from [Kaggle's challenge](https://www.kaggle.com/c/whats-cooking) (Go to _Data_ tab and click on _Download All_)
    -   Unzip and place `whats-cooking` directory inside the root directory of this project

### Repository structure
1. `cuisine-prediction.ipynb` notebook contains the full code along with the results obtained
2. If GitHub is unable to render the above notebook in your browser, you can instead download and see the notebook's equivalent HTML export `cuisine_prediction.html` (see `Exports/` directory)
3. `Submissions/` directory contains predictions of various models on Kaggle's test data (these can be submitted directly on Kaggle)
4. We split Kaggle's train data into `my_train_split.json` and `my_test_split.json` so that we can do more sophisticated analysis of results rather than just analyzing accuracy (this is necessary since ground truth of Kaggle's test data is not available)
5. `embeddings/` contains files that hold [node2vec](https://github.com/aditya-grover/node2vec) embeddings of the nodes of the `my_train_split` network

### Results summary
_Note:_ See the notebook for in-depth analysis of results

| Model                | Accuracy | Weighted F1-score | Unweighted F1-score |
|----------------------|----------|-------------------|---------------------|
| Baseline #1          | 53.2     | 0.459             | 0.268               |
| Baseline #2          | 40.92    | 0.319             | 0.140               |
| Baseline #2(b)       | 52.97    | 0.450             | 0.249               |
| NN (1-hot)           | 77.82    | 0.776             | 0.701               |
| SVM (1-hot)          | 76.71    | 0.763             | 0.684               |
| NN (embedding)       | 72.41    | 0.714             | 0.600               |
| SVM (embedding)      | 69.25    | 0.660             | 0.489               |
| GRU (embedding)      | 65.11    | 0.619             | 0.441               |
| Clustering heuristic | 40.03    | 0.382             | 0.246               |

### References
1. [What's Cooking](https://www.kaggle.com/c/whats-cooking) challenge at Kaggle
2. [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) by Aditya Grover and Jure Leskovec
3. Available implementation of [node2vec](https://github.com/aditya-grover/node2vec) by [Aditya Grover](https://github.com/aditya-grover)
