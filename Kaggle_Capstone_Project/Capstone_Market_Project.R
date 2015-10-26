library(readr)
library(xgboost)

set.seed(721)

cat("reading the train and test data\n")
train <- readr::read_csv("/Users/mingmingguo/Documents/market_train.csv")
test <- readr::read_csv("/Users/mingmingguo/Documents/test.csv")

train.unique.count=lapply(train, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const=names(train.unique.count_1)
delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

train=train[,!(names(train) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]

# Convert the date features to numerical features for train dataset
datecolumns = c("VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158", "VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176", "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0204", "VAR_0217")
train_cropped <- train[datecolumns]
train_cc <- data.frame(apply(train_cropped, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) #2 = columnwise

for (dc in datecolumns){
  train[dc] <- NULL
  train[dc] <- train_cc[dc]
}

train_cc <- NULL
train_cropped <- NULL
gc()

train_target <- train$target
train$target <- NULL
train$target <- train_target

# Do the same process for the test dataset
test_cropped <- test[datecolumns]
test_cc <- data.frame(apply(test_cropped, 2, function(x) as.double(strptime(x, format='%d%b%y:%H:%M:%S', tz="UTC")))) 

for (dc in datecolumns){
  test[dc] <- NULL
  test[dc] <- test_cc[dc]
}

test_cc <- NULL
test_cropped <- NULL
gc()


feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels = levels))
    test[[f]] <- as.integer(factor(test[[f]], levels = levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)] <- -1

cat("sampling train to get around 8GB memory limitations\n")
train <- train[sample(nrow(train), 120000),]
gc()

samp = sample(nrow(train), 80000)

train <- train[samp,]
gc()

valid <- train[-samp,]
gc()


# Train a XGBoost Classifier
cat("training a XGBoost Classifier\n")
dtrain <- xgb.DMatrix(data.matrix(train[,feature.names]), label=train$target)
dvalid <- xgb.DMatrix(data.matrix(valid[,feature.names]), label=valid$target)

watchlist <- watchlist <- list(eval = dvalid, train = dtrain)

param <- list(objective           = "binary:logistic",
              eval_metric         = "auc",
              eta                 = 0.02, 
              max_depth           = 8,
              subsample           = 0.7, 
              colsample_bytree    = 0.8
              )

clf <- xgb.train(params      = param,
                 data        = dtrain,
                 nrounds     = 800,
                 verbose     = 1,
                 watchlist   = watchlist,
                 maximize    = TRUE)

dtrain <- 0
gc()

dvalid <- 0
gc()


cat("making predictions in batchs due to 8GB memory limitation\n")
submission <- data.frame(ID=test$ID)
submission$target <- NA
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(test[rows, feature.names]))
}


cat("saving the submission file\n")
readr::write_csv(submission, "/Users/mingmingguo/Documents/xgboost_submission.csv")