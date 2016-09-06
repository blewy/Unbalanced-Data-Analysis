library(xgboost)
library(Matrix)
library(caret)
library(Amelia)
library(readr)
library(magrittr)
library(data.table)
library(tidyr) # Tidy the dataset.
library(ggplot2) # Visualise data.
library(dplyr) # Data preparation and pipes %>%.
#library(FSelector) # Feature selection.
library(corrplot)
library(pROC)
options(scipen=999)

ls()

# ---------------------------------------------------
# Load
orig.train <- read.csv("./data/train.csv", stringsAsFactors = F)

prop.table(table(orig.train$TARGET))

orig.train$TARGET<- factor(orig.train$TARGET, labels = c("No", "Yes"))

target<- factor(orig.train$TARGET, labels = c("No", "Yes"))

prop.table(table(target))

#Take a look
orig.train[sample(nrow(orig.train),6),]

#structure
str(orig.train)

#dimension
dim(orig.train)

# Classe od the variable 
sapply(orig.train, class)

# Setting the variable ROles 

(vars <- names(orig.train))
target <- "TARGET"
id <- "ID"

# Lets sett-up the variable to ignore

ignore <- id

#lookinh for variable thar mostly serve as id
(ids <- which(sapply(orig.train, function(x) length(unique(x))) == nrow(orig.train)))
#only the ID, that good

ignore <- union(ignore, names(ids))

# All Missing (missing value count - mvc)
mvc <- sapply(orig.train[vars], function(x) sum(is.na(x)))
#all values missing
mvn <- names(which(mvc == nrow(orig.train)))
mvn
#not one 
ignore <- union(ignore, mvn)

#Many Missing 
mvn <- names(which(mvc >= 0.7*nrow(orig.train)))
mvn #not one
ignore <- union(ignore, mvn)

#Too Many Levels
factors <- which(sapply(orig.train[vars], is.factor))
lvls <- sapply(factors, function(x) length(levels(orig.train[[x]])))
(many <- names(which(lvls > 20)))

ignore <- union(ignore, many)
#constants variables

(constants <- names(which(sapply(orig.train[vars], function(x) all(x == x[1L])))))
ignore <- union(ignore, constants)

##### Removing identical features
features_pair <- combn(names(orig.train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(orig.train[[f1]] == orig.train[[f2]])) {
      # cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

ignore <- union(ignore, toRemove)


### Remove features with less than 10% of non-zero entries
zero.rate <- sapply(orig.train[vars], function(dt.col) {
  sum(dt.col == 0)/length(dt.col)
})

(non_info <- vars[zero.rate > 0.9])

ignore <- union(ignore, non_info)



vars_final <- setdiff(vars, ignore)

data.test<-orig.train[,vars_final]
# Feature engeniering

# replace with most common
data.test$var3[data.test$var3==-999999] <- 2
data.test$var38 <- log(data.test$var38)

save(data.test, file='data.test.rda') 

target<-as.factor(data.test$TARGET)
train_matrix <- as.data.frame(model.matrix( ~ .-1, data = data.test[,-63], sparse=FALSE))

names(train_matrix)


set.seed(156)

split1 <- createDataPartition(target, p = .7)[[1]]
other     <- train_matrix[-split1,]
other_target <-target[-split1]
training  <- train_matrix[split1,]
training_target <- target[split1]


set.seed(934)

split2 <- createDataPartition(other_target, p = 1/3)[[1]]
evaluation  <- other[ split2,]
evaluation_target <- other_target[split2]
testing     <- other[-split2,]
testing_target <- other_target[-split2]

### These functions are used to measure performance

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
fourStats <- function (data, lev = levels(data$obs), model = NULL)
{
  
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}

ctrl <- trainControl(method = "cv",
                     number=3,
                     classProbs = TRUE,
                     summaryFunction = fiveStats)

ctrlNoProb <- ctrl
ctrlNoProb$summaryFunction <- fourStats
ctrlNoProb$classProbs <- FALSE

ctrl$sampling <- NULL # No sampling

#number of tree 
tress_n<-25

mtry_grid <- data.frame(mtry = c(1:15, (4:9)*5))

set.seed(1401)
rf = train( y=training_target , 
            x=training,
            method = "rf", 
            ntree = tress_n,
            metric = "ROC",
            tuneGrid=mtry_grid,
            trControl = ctrl)
rf
plot(rf)


# Internal down-sampling

set.seed(1537)
rf_down_int <- train(y=training_target , 
                     x=training,
                     method = "rf",
                     metric = "ROC",
                     strata = training_target,
                     sampsize = rep(sum(training_target == "Yes"), 2),
                     ntree = tress_n,
                     tuneGrid = mtry_grid,
                     trControl = ctrl)
rf_down_int
plot(rf_down_int)

## External down-sampling

ctrl$sampling <- "down"
set.seed(1537)

set.seed(1537)
rf_down_ext <- train(y=training_target , 
                     x=training,
                     method = "rf",
                     metric = "ROC",
                     ntree = tress_n,
                     tuneGrid = mtry_grid,
                     trControl = ctrl)
rf_down_ext
plot(rf_down_ext)


## External down-sampling

ctrl$sampling <- "up"

set.seed(1537)
rf_up_ext <- train(y=training_target , 
                     x=training,
                     method = "rf",
                     metric = "ROC",
                     ntree = tress_n,
                     tuneGrid = mtry_grid,
                     trControl = ctrl)
rf_up_ext
plot(rf_up_ext)

## External SMOOT

ctrl$sampling <- "smote"

set.seed(1537)
rf_smote <- train(y=training_target , 
                   x=training,
                   method = "rf",
                   metric = "ROC",
                   ntree = tress_n,
                   tuneGrid = mtry_grid,
                   trControl = ctrl)
rf_smote
plot(rf_smote)

## External SMOOT

ctrl$sampling <- "rose"

set.seed(1537)
rf_rose <- train(y=training_target , 
                  x=training,
                  method = "rf",
                  metric = "ROC",
                  ntree = tress_n,
                  tuneGrid = mtry_grid,
                  trControl = ctrl)
rf_rose
plot(rf_rose)


## External SMOTE with more neighbors! - User defines sammpling

smotest <- list(name = "SMOTE with more neighbors!",
                func = function (x, y) {
                  library(DMwR)
                  dat <- if (is.data.frame(x)) x else as.data.frame(x)
                  dat$.y <- y
                  dat <- SMOTE(.y ~ ., data = dat, k = 10)
                  list(x = dat[, !grepl(".y", colnames(dat), fixed = TRUE)],
                       y = dat$.y)
                },
                first = TRUE)



ctrlSmoteTest <- trainControl(method = "cv",
                     number=3,
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     sampling = smotest)

set.seed(1537)
rf_smotest <- train(y=training_target , 
                 x=training,
                 method = "rf",
                 metric = "ROC",
                 ntree = tress_n,
                 tuneGrid = mtry_grid,
                 trControl = ctrlSmoteTest)
rf_smotest
plot(rf_smotest)

## External ROSE with more neighbors! - User defines sammpling


ROSEtest <- list(name = "ROSE with Artificail data Shrinkage!",
                func = function (x, y) {
                  library(ROSE)
                  dat <- if (is.data.frame(x)) x else as.data.frame(x)
                  dat$.y <- y
                  dat <- ROSE(.y ~ ., data = dat, seed = 1, hmult.majo = 0.25,
                              hmult.mino = 0.5)$data
                  list(x = dat[, !grepl(".y", colnames(dat), fixed = TRUE)],
                       y = dat$.y)
                },
                first = TRUE)


ctrlRoseTest <- trainControl(method = "cv",
                     number=3,
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     sampling = ROSEtest)

set.seed(1537)
rf_rosetest <- train(y=training_target , 
                    x=training,
                    method = "rf",
                    metric = "ROC",
                    ntree = tress_n,
                    tuneGrid = mtry_grid,
                    trControl = ctrlRoseTest)
rf_rosetest
plot(rf_rosetest)



# Summayu --- 

library(pROC)


samplingSummary <- function(x, evl, tst)
{
  lvl <- rev(levels(testing_target))
  evlROC <- roc(evaluation_target,
                predict(x, evl, type = "prob")[,1],
                levels = lvl)
  tstROC <- roc(evaluation_target,
                predict(x, evl, type = "prob")[,1],
                levels = lvl)
  rocs <- c(auc(evlROC), auc(tstROC))
  cut <- coords(evlROC, x = "best", ret="threshold",
                best.method="closest.topleft")
  bestVals <- coords(tstROC, cut, ret=c("sensitivity", "specificity"))
  out <- c(rocs, bestVals*100,cut)
  names(out) <- c("evROC", "tsROC", "tsSens", "tsSpec","closest.topleft Treshold")
  out
  
}


samplingSummary(rf, evaluation, testing)

rfResults <- rbind(samplingSummary(rf, evaluation, testing),
                   samplingSummary(rf_down_int, evaluation, testing),
                   samplingSummary(rf_down_ext, evaluation, testing),
                   samplingSummary(rf_up_ext, evaluation, testing),
                   samplingSummary(rf_smote, evaluation, testing),
                   samplingSummary(rf_rose, evaluation, testing),
                   samplingSummary(rf_smotest, evaluation, testing),
                   samplingSummary(rf_rosetest, evaluation, testing))
rownames(rfResults) <- c("Original", "Down Sampling (Internal)",  "Down Sampling", "UP Sampling", "SMOTE","ROSE","SMOTE TEST","Rose TEST")

rfResults



### --------  Gradient Bosting Trees (GBM) -------------

ctrl <- trainControl(method = "cv",
                     number=3,
                     classProbs = TRUE,
                     summaryFunction = fiveStats,
                     sampling <- NULL)# No sampling



# gbmGrid <-  expand.grid(interaction.depth = c(10,15,25),
#                         n.trees = (1:3)*75,
#                         shrinkage = c(0.05,0.1),
#                         n.minobsinnode = c(20,30))

cost_grid <- expand.grid(trials = 1:5,
                         winnow = FALSE,
                         model = "tree",
                         cost = seq(1, 10, by = 1))

set.seed(998)
gbmFit <- train(y=training_target , 
                x=training,
                method = "C5.0",
                verbose = FALSE,
                tuneGrid = gbmGrid,
                metric = "ROC",
                trControl = ctrl)
gbmFit
plot(gbmFit)


class.freq <- table(training_target) %>% prop.table
row.wt <- ifelse(training_target == 'No', 1/class.freq[1], 1/class.freq[2])

gbmFit.wt <- train(y=training_target , 
                x=training,
                method = "gbm",
                verbose = FALSE,
                weights = row.wt,
                tuneGrid = gbmGrid,
                metric = "ROC",
                trControl = ctrl)
gbmFit.wt
plot(gbmFit.wt)

samplingSummary(gbmFit, evaluation, testing)
samplingSummary(gbmFit.wt, evaluation, testing)



### -------- 5.0 cost------------

ctrl_cost <- trainControl(method = "repeatedcv",
                          repeats = 3,
                          classProbs = FALSE,
                          savePredictions = TRUE,
                          summaryFunction = fourStats)


# cost_grid <- expand.grid(trials = 1:5,
#                          winnow = FALSE,
#                          model = "tree",
#                          cost = seq(1, 10, by = 1))

c5Grid <- expand.grid(trials = 5,
                         winnow = FALSE,
                         model = "tree",
                         cost = 10)


c5Fit.cost <- train(y=training_target , 
                   x=training,
                   method = "C5.0Cost",
                   verbose = FALSE,
                   tuneGrid = c5Grid,
                   metric = "Kappa",
                   trControl = ctrl_cost)
c5Fit.cost


##  SVM ------------

library( kernlab)
data.sigeste <-cbind(training_target,training)
srange<- sigest(training_target~.,data = data.sigeste)

class.freq <- table(training_target) %>% prop.table


svmRGridReduced <- expand.grid(sigma = srange[2], C = 2^c(2), Weight=1:2)

set.seed(998)
svmRModel <- train(y= training_target,
                   x= training ,
                   method = "svmRadialWeights", 
                   tuneGrid = svmRGridReduced, 
                   metric = "kappa",
                   trControl = ctrl_cost)
svmRModel
plot(svmRModel)

# 
# eXtreme Gradient Boosting
# 
# method = 'xgbTree'
# 
# Type: Regression, Classification
# 
# Tuning Parameters: nrounds (# Boosting Iterations), max_depth (Max Tree Depth), eta (Shrinkage), gamma (Minimum Loss Reduction), colsample_bytree (Subsample Ratio of Columns), min_child_weight (Minimum Sum of Instance Weight)
