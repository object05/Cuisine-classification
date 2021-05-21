# Overview of Data 
# Check working directory
getwd()
# Import JSON files into R
install.packages("rjson")
library(rjson)
train_raw <- fromJSON(file = "train.json")
test_raw <- fromJSON(file = "test.json")
# Overview
str(train_raw)
str(test_raw)
# Plot the train set to check frequency
dat <- lapply(train_raw, function(j) {
  as.data.frame(replace(j, sapply(j, is.list), NA))
})
library(plyr)
res <- rbind.fill(dat)
ggplot(data = res, aes(x = cuisine)) + 
  geom_histogram() +
  labs(title = "Cuisines", x = "Cuisine", y = "Number of Recipes")

# Decision Tree 
library(class)
library(caret)
library(caTools)
library(tm)
library(rpart)
library(rpart.plot)
library(data.table)
library(Matrix)
library(SnowballC)

# load data files and flatten
train_raw  <- fromJSON("../input/train.json", flatten = TRUE)
test_raw <- fromJSON("../input/test.json", flatten = TRUE)

# process the ingredients data for model
# convert upper-case characters in a character vector to lower-case
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=tolower)
test_raw$ingredients <- lapply(test_raw$ingredients, FUN=tolower)

# eliminate dash -/_
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("-", "_", x)) 
test_raw$ingredients <- lapply(test_raw$ingredients, FUN=function(x) gsub("-", "_", x))

# eliminate regular character and spaces
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) 
test_raw$ingredients <- lapply(test_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x))

# create a matrix of ingredients in both the TRAIN and TEST set
my_ingredients<-c(Corpus(VectorSource(train_raw$ingredients)), Corpus(VectorSource(test_raw$ingredients)))

# create simple document term matrix
my_ingredientsDTM <- DocumentTermMatrix(my_ingredients)
# remove sparse terms 
my_ingredientsDTM <- removeSparseTerms(my_ingredientsDTM, 0.995) 
# change out term document matrix to a data frame
my_ingredientsDTM <- as.data.frame(as.matrix(my_ingredientsDTM))

# add simple feature: count of ingredients per receipe
my_ingredientsDTM$ingredients_count  <- rowSums(my_ingredientsDTM) 

# add cuisine for TRAIN set, default to "italian" for the TEST set
my_ingredientsDTM$cuisine <- as.factor(c(train_raw$cuisine, rep("italian", nrow(test_raw))))

# split the DTM into TRAIN and TEST sets
dtm_train  <- my_ingredientsDTM[1:nrow(train_raw), ]
dtm_test <- my_ingredientsDTM[-(1:nrow(train_raw)), ]

# Construct decision tree to predict
predict<-rpart(cuisine~., data=dtm_train, method="class") 
# Plot the rpart model (decision tree) by labelling all nodes 
# and displaying the probability per class of observations in the node
prp(predict, type=1, extra=4)
prp(predict)

# build and write the submission file
test_predict <- predict(predict, newdata = dtm_test, type = "class")
submission1 <- cbind(as.data.frame(test_raw$id), as.data.frame(test_predict))
colnames(submission1) <-c("id","cuisine")
write.csv(submission1, file = 'DecisionTree.csv', row.names=F, quote=F)
 
# Add Xgboost package 




# prepare the spare matrix 
xgbmat<- xgb.DMatrix(Matrix(data.matrix(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])), label=as.numeric(dtm_train$cuisine)-1)

# train our multiclass classification model using softmax
xgb<- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)

# predict on the TEST set and change cuisine back to string
xgb.test <- predict(xgb, newdata = data.matrix(dtm_test[, !colnames(dtm_test) %in% c("cuisine")]))
xgb.test.text <- levels(dtm_train$cuisine)[xgb.test+1]

# load sample submission file to use as a template for submission
sample_sub <- read.csv('../input/sample_submission.csv')

# build and write the submission file
test_match <- cbind(as.data.frame(test_raw$id), as.data.frame(xgb.test.text))
colnames(test_match) <- c("id", "cuisine")
test_match   <- data.table(test_match, key="id")
test_cuisine <- test_match[id==sample_sub$id, as.matrix(test_match$cuisine)]

submission2 <- data.frame(id = sample_sub$id, cuisine = test_cuisine)
write.csv(submission2, file = 'RXgboost.csv', row.names=F, quote=F)