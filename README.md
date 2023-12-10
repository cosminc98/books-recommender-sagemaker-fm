# Book Recommender System with Explicit Feedback

1. [Introduction](#introduction)
    * [Dataset](#dataset)
    * [Definitions](#definitions)
    * [Factorization Machines](#factorization-machines)
    * [Anonymous Features Generation](#anonymous-features-generation)
    * [Reducing the Search Space](#reducing-the-search-space)
2. [Project Structure](#project-structure)
3. [Results](#results)
    * [Model Evaluation](#model-evaluation)
    * [Example Recommendation](#example-recommendation)
4. [Pretrained Model](#pretrained-model)

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
    * Here you can deploy the factorization machine model using a SageMaker endpoint.
    * You can also get book recommendations in under 1 second using the proximal books to reduce the search space and the factorization
machine model to rank them.

<a name="results"></a>
## Results

<a name="model-evaluation"></a>
### Model Evaluation

We evaluate the model and a baseline on a separate subset than the one we trained on. The test subset does not overlap with the training subset in terms of users.
The baseline just assigns the average rating for that book from the training subset while completely ignoring the context books. Figure 2 shows the distribution of
the true ratings in the first N samples from the test subset sorted by the predicted rating from our model and from the baseline:

<div style="width:50%;margin-left:auto;margin-right:auto;">
    <figure>
        <img width="750" src="https://cosminc98-public-models.s3.eu-central-1.amazonaws.com/books-recommender/factorization-machine/ratings_distribution.png"></br>
        <figcaption style="text-align:left">
          <b>Figure 2.</b>
        </figcaption>
    </figure>
</div>

We draw the following conclusions:
* The percentage of terrible recommendations (with true rating of 0) decreases from 40% using the baseline to 10% using our model for top 25, and from 35% to 18% for top 100.
* The percentage of excellent recommendations (with true rating of 10) increases from 24% using the baseline to 43% using our model for top 25, and from 25% to 44% for top 100.
* This means that our model, while not achieving a great improvement in mean absolute error compared to assigning the mean rating to every sample, is capable of assigning slightly higher predicted ratings to books that were actually good, which is all that we care about when we are ranking samples to recommend. Our model had a mean absolute
error of 0.383 and the baseline of 0.405 (the predicted ratings were normalized between 0 and 1).

In Figure 3 we see almost the same information, but we binarize the true book ratings into "Relevant" or "Not Relevant", where "Relevant" means a rating of at least 7.
With our model, we have 68% relevant samples vs. 60% with the baseline in the top 25 samples. For the top 100 samples, our model has 75% relevant samples and the baseline
only 59%.

<div style="width:50%;margin-left:auto;margin-right:auto;">
    <figure>
        <img width="750" src="https://cosminc98-public-models.s3.eu-central-1.amazonaws.com/books-recommender/factorization-machine/relevance_distribution.png"></br>
        <figcaption style="text-align:left">
          <b>Figure 3.</b>
        </figcaption>
    </figure>
</div>

In conclusion, not only does the factorization machine model provide more relevant samples, but both the relevant and the irrelevant samples have higher ratings overall.

<a name="example-recommendation"></a>
### Example Recommendation

Here is an example of book recommendations from 3 context books. You will be able to see the main problem with the model at the moment, which is left for future work.
The problem is the imperfect deduplication, which leads the model to recommend the same book as the one we told it we just read. Although it is the same book, they are
two versions, with two different ISBNs.

The books that the recommendations were based on:
* "The Selfish Gene" by "Richard Dawkins"
* "Climbing Mount Improbable" by "Richard Dawkins"
* "A Clash of Kings" by "George R.R. Martin"

The recommended books:
* "A Clash of Kings" by "George R.R. Martin"
* "Windhaven" by "George R.R. Martin"
* "Warchild" by "Karin Lowachee"
* "A Storm of Swords" by "George R.R. Martin"
* "The Blind Watchmaker: Why the Evidenc..." by "Richard Dawkins"
* "The Biotech Century: Harnessing the G..." by "Jeremy Rifkin"
* "Shock" by "Robin Cook"
* "Battlefield Earth: A Saga of the Year..." by "L. Ron Hubbard"
* "Sarajevo Daily: A City and Its Newspa..." by "Tom Gjelten"
* "My Century: A Novel" by "Gunter Grass"
* "Fevre Dream" by "George R.R. Martin"
* "Tuf Voyaging" by "George R.R. Martin"
* "The Extended Phenotype: The Long Reac..." by "Richard Dawkins"
* "Joker's Wild (Wild Cards, Vol 3)" by "George R.R. Martin"
* "Down and Dirty (Wild Cards, No 5)" by "George R.R. Martin"
* "The Green Lifestyle Handbook: 1001 Wa..." by "Jeremy Rifkin"
* "The Danzig Trilogy: The Tin Drum, Cat..." by "Gunter Grass"
* "Der entzauberte Regenbogen. Wissensch..." by "Richard Dawkins"
* "Wild Cards (Volume 1)" by "George R.R. Martin"
* "Blood of the Fold (Sword of Truth, Bo..." by "Terry Goodkind"

<a name="pretrained-model"></a>
## Pretrained Model

You require multiple files to run the inference notebook:
1. The factorization machine model parameters which you need not download as SageMaker can deploy an endpoint using the S3 object at [this URL](https://cosminc98-public-models.s3.eu-central-1.amazonaws.com/books-recommender/factorization-machine/model.tar.gz).
2. The other files are contained in [this archive](https://cosminc98-public-models.s3.eu-central-1.amazonaws.com/books-recommender/factorization-machine/preprocessing_assets.zip), and you need to have it available on the local file system of your notebook instance.
