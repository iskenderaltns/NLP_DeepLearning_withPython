### NLP_DeepLearning_withPython
CLASSIFICATION OF CUSTOMER REVIEWS
## **Approach For the Classification of Comments**
- **Data preparation:** I will use it to clean and preprocess the data in the data. For example, I will do cleanup and tokenization to make comments machine understandable. Also, I need to parse the data for training and testing.
- **Train-test split:** One method I can use to parse data for training and testing is "train-test split". This method decomposes the data as training and test data in a certain ratio. For example, I can use 80% of the data for training and the remaining 20% for testing.
- **Model integration:** Next, I will create a text classification model. This model takes product comments as input and outputs which category the products belong to. For example, an LSTM (Long Short-Term Memory) or a transformers model can be used.
- **Training and testing the model:** I will use the training data to train the model I created. Next, I will measure the performance of the model using the test data.

## **Experiments Data:**
This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its nine supportive features offer a great environment to parse out the text through its multiple dimensions. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.

This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:

- **Clothing ID:** Integer Categorical variable that refers to the specific piece being reviewed.
- **Age:** Positive Integer variable of the reviewer’s age.
- **Title:** String variable for the title of the review.
- **Review Text:** String variable for the review body.
- **Rating:** Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst to 5 Best.
- **Recommended IND:** Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
- **Positive Feedback Count:** Positive Integer documenting the number of other customers who found this review positive.
- **Division Name:** Categorical name of the product high level division.
- **Department Name:** Categorical name of the product department name.
- **Class Name:** Categorical name of the product class name.


## **Text Preprocessing**
While doing Text preprocessing we started with Tokenization, Punctuation removal, Stop word removal, numbers removal, lowering the text.

## **Stemming**
I will use Lancaster Stemmer instead of Porter Stemmer here.

## **Vectorization/ Model Building/**

- *Train-test split:* I will use 80% of the data for training and the remaining 20% for testing.
- *Embedding:* GloVe50d uses 50-dimensional vectors to represent each word, while GloVe300d uses 300- dimensional vectors. This means that GloVe300d's vectors have more "degrees of freedom" to represent words and can therefore capture finer relationships between words. 
- *Model integration:* Next, I will create a text classification model. This model will take product comments as input and output which category the products belong to.
## **Model** 
First, an input layer named int\_sequences\_input is created. This layer receives data according to the size determined by the shape parameter. It accepts the data type specified by the dtype parameter. Then, with the embedding\_layer function, the input data is translated as embedded sequences. This process transforms the data into a vector form. Next, the data is normalized with the layers.BatchNormalization function. This is standardization of data. Next, the data is passed to a convolutional layer with the layers.Conv1D function. This layer filters the data and extracts its properties.Next, the data is dropped with the layers.Dropout function. This action prevents the model from overfitting. Next, the data is max pooled with the function layers.MaxPooling1D. This process further extracts the characteristics of the data. These processes are repeated and finally the global max pooling is done with the data layers.GlobalMaxPooling1D() function. Next, the data is passed to the dense layer with the layers.Dense function. This layer uses data to classify. Finally, the data is dropped with the layers.Dropout function. At the end of these processes, the text classification model is created and can guess which category the product reviews belong to. I choose to run for 25 epochs.

=============================================================================================================
- *Epoch 20/25*
-- 394/394 - 181s - loss: 0.8406 - acc: 0.7020 - val\_loss: 4.0405 - val\_acc: 0.4650 - 181s/epoch - 460ms/step
- *Epoch 21/25*
-- 394/394 - 182s - loss: 0.8187 - acc: 0.7102 - val\_loss: 2.6071 - val\_acc: 0.5118 - 182s/epoch - 462ms/step
- *Epoch 22/25*
-- 394/394 - 182s - loss: 0.8031 - acc: 0.7173 - val\_loss: 3.4224 - val\_acc: 0.5181 - 182s/epoch - 461ms/step
- *Epoch 23/25*
-- 394/394 - 182s - loss: 0.7875 - acc: 0.7225 - val\_loss: 3.1251 - val\_acc: 0.5099 - 182s/epoch - 462ms/step
- *Epoch 24/25*
-- 394/394 - 182s - loss: 0.7753 - acc: 0.7282 - val\_loss: 2.8715 - val\_acc: 0.4984 - 182s/epoch - 462ms/step
- *Epoch 25/25*
-- 394/394 - 182s - loss: 0.7499 - acc: 0.7382 - val\_loss: 2.3719 - val\_acc: 0.5229 - 182s/epoch - 461ms/step


We see that the loss is dropping for each epoch and the accuracies are rising for the train data, as expected. We seem to eventually run into overfitting for the validation data, though this was of course expected, and the early stopping object takes care of that problem. Different runs yielded different values across epochs; however, the qualitative result was always the same (first improving validation loss and accuracy, then worsening). The best accuracy score we get is 73.82%, so our model is better than the baseline of 50.


#**NOTE!!**
Categorizing product reviews by “class” did not work well. Now let's try to categorize the product reviews according to their “department” with the same model.


- *Epoch 20/25*
-- 394/394 - 164s - loss: 0.2648 - acc: 0.9150 - val\_loss: 2.5037 - val\_acc: 0.7228 - 164s/epoch - 417ms/step
- *Epoch 21/25*
-- 394/394 - 166s - loss: 0.2563 - acc: 0.9170 - val\_loss: 5.7951 - val\_acc: 0.5642 - 166s/epoch - 422ms/step
- *Epoch 22/25*
-- 394/394 - 166s - loss: 0.2437 - acc: 0.9212 - val\_loss: 4.1979 - val\_acc: 0.6224 - 166s/epoch - 420ms/step
- *Epoch 23/25*
-- 394/394 - 165s - loss: 0.2282 - acc: 0.9276 - val\_loss: 3.3032 - val\_acc: 0.6783 - 165s/epoch - 419ms/step
- *Epoch 24/25*
-- 394/394 - 160s - loss: 0.2283 - acc: 0.9272 - val\_loss: 3.5183 - val\_acc: 0.6529 - 160s/epoch - 407ms/step
- *Epoch 25/25*
-- 394/394 - 159s - loss: 0.2203 - acc: 0.9273 - val\_loss: 4.5338 - val\_acc: 0.6796 - 159s/epoch - 404ms/step

Categorizing products by departments worked better than I expected. The best accuracy score we get is 92.73%, so our model is better than the baseline of 50%.



------------------------------------------------

In the previous version we had 2256 misses. This may show why it is not a good idea not to apply stemming.

# Converted 4785 words (2256 misses)

When you stem a word, say "leaving" or "studies" it is converted into "leav" or "studi". After stemming the new token does not need to be a true word, that is why, it may not be represented in a pretrained model. As a conclusion, it is wise to use lemmatization instead of stemming here.
however, when we use lemmatisation instead of stemming we get:

# Converted 9196 words (1349 misses)

Now let’s try to categorize the product reviews according to their “department” with the same model using lemmatisation.

- *Epoch 20/25*
-- 394/394 - 190s - loss: 0.2176 - acc: 0.9315 - val\_loss: 1.0109 - val\_acc: 0.8010 - 190s/epoch - 481ms/step
- *Epoch 21/25*
-- 394/394 - 190s - loss: 0.2159 - acc: 0.9340 - val\_loss: 1.2864 - val\_acc: 0.7985 - 190s/epoch - 482ms/step
- *Epoch 22/25*
-- 394/394 - 189s - loss: 0.1974 - acc: 0.9378 - val\_loss: 1.2444 - val\_acc: 0.8074 - 189s/epoch - 480ms/step
- *Epoch 23/25*
-- 394/394 - 190s - loss: 0.1959 - acc: 0.9376 - val\_loss: 1.2708 - val\_acc: 0.8090 - 190s/epoch - 483ms/step
- *Epoch 24/25*
-- 394/394 - 187s - loss: 0.1945 - acc: 0.9394 - val\_loss: 1.0969 - val\_acc: 0.7734 - 187s/epoch - 476ms/step
- *Epoch 25/25*
-- 394/394 - 188s - loss: 0.1846 - acc: 0.9412 - val\_loss: 1.0160 - val\_acc: 0.8258 - 188s/epoch - 478ms/step

The best accuracy score we get is 94.12%, so our model is better than the baseline of 50%. this is almost a better result than the previous one.


## **Data Augmentation**
One additional way to reduce the risk of overfitting is to apply data augmentation. In the data we used, the “Positive Feedback Count” part indicates the number of people who found the comment to be correct. This means that we can add the same comment as “Positive Feedback” number of times for that product. Thus, we increase our data. this one is just a suggestion. will not be applied.
