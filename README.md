# Book Recommender System with Explicit Feedback

1. [Introduction](#introduction)
    * [Dataset](#dataset)
    * [Definitions](#definitions)
    * [Factorization Machines](#factorization-machines)
    * [Anonymous Features Generation](#anonymous-aeatures-generation)
    * [Reducing the Search Space](#reducing-the-search-space)
2. [Project Structure](#project-structure)
3. [Results](#results)
    * [Model Evaluation](#nodel-evaluation)
    * [Example Recommendation](#example-recommendation)

<a name="introduction"></a>
## Introduction
---

This project uses factorization machines to build a recommender system that can handle anonymous users (which the model has not seen during training) by
simply having them provide a few books that they read and liked. [This](https://github.com/edrans/-aws-sagemaker-builtin-notebooks/blob/main/factorization-machine/Explicit-feedback/Recommendation-Machine-Explicit.ipynb) 
notebook was used as a starting point for training a factorization machine in AWS SageMaker. The main differences between that one and this project are in 
data preprocessing, having eliminated the user features to allow for anonymous inference without retraining, and in changing the factorization machine from 
binary classification to regression. The reason for changing to regression is trying to obtain as many "10/10" rated books (true labels) in the top 25 / top 100 
books sorted by the factorization machine predicted ratings. If using binary classification, either we'll have too few books for the positive class by choosing 
only the rating "10" as relevant, or we'll be unable to differentiate between 10's, 9's, 8's and 7's if we choose relevance to mean a rating of at least 7.

<a name="dataset"></a>
### Dataset
The dataset we use is from [here](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/). However, I preprocessed it using SageMaker Data Wrangler. 
The main preprocessing step was creating a train/dev/test split for the user ratings in such a way as to not have the same user in multiple subsets. To download 
the preprocessed data click [here](https://cosminc98-public-datasets.s3.eu-central-1.amazonaws.com/books-recommender/all.zip).

<a name="definitions"></a>
### Definitions
| Term | Definition |
|:--- |:--- | 
| Context Book | A book that is provided by the user, with the assumption that they liked it (explicit feedback), that will be used to rank recommendations using the factorization machine model. The context books are only a subset of the book dataset since some books did not have enough data during training. **Users may only choose liked books from this subset.** |
| Target Book | A book that can be scored by the factorization machine model. The target books are only a subset of the book dataset since some books did not have enough data during training. **Users will receive recommendations only from this subset.** |
| Proximal Book | A context book has multiple proximal books that are "close" to it. Here, "close" means that users that liked the context book also liked the proximal books. Books from the same authors are also considered proximal. We use proximal books to reduce the search space. |
| ISBN | International Standard Book Numbers (or ISBN) is a unique identified for each book. In particular, it is an identified for each specific version/revision of a given book, which is why we need to perform ISBN deduplication to map old versions to the latest one. We do this to treat all versions of the same book in as one single book. |

<a name="factorization-machines"></a>
### Factorization Machines (FM)

The method we will use is a Factorization Machines regressor. FM is a general-purpose supervised learning algorithm that you can use for classification and regression 
tasks. It excels for extremely sparse datasets such as the one that we're dealing with since the average book has only 3 ratings and 70% of users have rated 3 books 
or fewer. The Amazon SageMaker FM algorithm provides a robust and highly scalable implementation of this algorithm.

<div style="width:50%;margin-left:auto;margin-right:auto;">
    <figure>
        <img style="width:100%" src="https://cosminc98-public-models.s3.eu-central-1.amazonaws.com/books-recommender/factorization-machine/anonymous_features.svg">
        <figcaption style="text-align:left">
          <b>Figure 1 - Anonymous Features</b>; they are anonymous because we do not one-hot encode the user ID and instead we rely on the interactions between 
          the book that we want to rate, called target books (TB), and the books that the anonymous user said they read and liked, called context books (CB), 
          which are provided via explicit feedback, to accurately predict if an anonymous user will like a book. For more information on how these features are
          used see the Training notebook.
        </figcaption>
    </figure>
</div>
</br>

<a name="anonymous-features-generation"></a>
### Anonymous Features Generation

In our application we will have anonymous users provide a few books that they liked and we need to give them book recommendations based on those. 
Because of this, we will not use users as a feature. As we can see in Figure 1, we need tuples of 3 elements in order to perform training:
1. Target Book ISBN
2. Target Book Rating (needed for training but not for inference)
3. List of Context Books ISBNs

The problem is that our dataset provides us with tuples of the following 3 elements:
1. Target Book ISBN
2. Target Book Rating
3. User ID

In order to transform the latter into the former, we need to group all the ratings of each user and choose combinations of "liked books" 
(books with ratings >= 7, for example) and use them as context for another book that wasn't chosen this iteration to be in the context list.

<a name="reducing-the-search-space"></a>
### Reducing the Search Space

The dataset that our model is trained on contains around 200 000 books. Running our model on all of them would not only be computationally 
expensive, but it would also introduce a lot of noise since we would only use a few books that have the highest predicted rating and a few
books for which the rating is accidentally predicted very high will ruin the recommendations.

To get around this we reduce the search space by statically computing a vicinity around each context book (by following links in the ratings
graph). Each context book is assigned at most 100 such books, which we call "proximal" books. When a new user wants book recommendations we
can either only look at the proximal books of the context books they provided, or we can iteratively score the proximal books, rank them,
and then get the proximal books of the best proximal books to enlargen the search space however much we desire.

<a name="project-structure"></a>
## Project Structure

This project is divided in three notebooks. If you want to train a model from scratch, you should follow them in this order:
1. Training Notebook
    * Here you can train and evaluate a new factorization machine model that predicts book ratings given a few context books.
    * You can find detailed explanations on how the factorization machine algorithm works and how we preprocess the data.
2. Proximity Notebook
    * Here you can generate a list of proximal books, which are books that are similar to a given book, for every possible context book.
    * Proximal books are found by following what other books were liked by users that liked the given context book.
Books written by the same author are also considered proximal.
3. Inference Notebook
   * 
