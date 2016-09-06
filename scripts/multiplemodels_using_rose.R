#setwd("F:\\UTD material\\Kaggle")
set.seed(1234)
train = read.csv("../input/train.csv")


train$ID <- NULL
Const_vars <- names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0.000)])
train <- train[setdiff(names(train),Const_vars)]

#replacing null with NA values
train = replace(train,NULL,NA)

# Creating a correlation matrix
Corr_Mat <- cor(train)

#Replacing the upper triagle of the matrix with zero values to remove 
#the duplicate correlatons and for determining the variables with high colinearity
Corr_Mat[upper.tri(Corr_Mat)] <- 0 

#Setting diagonal values in the matrix to 0
diag(Corr_Mat) <- 0

#Removing all columns with value more than 0.90------CountVars = 169, 167 vars removed
Corr_Mat <- Corr_Mat[, apply(Corr_Mat,2,function(x) all(x<=0.92))]

#Removing all columns with value less than -0.90
Corr_Mat <- Corr_Mat[, apply(Corr_Mat,2,function(x) all(x>=-0.80))]

#Post correlation analysis, number of vriables reduced from 371 to 204

non_correlated_vars <- colnames(Corr_Mat)

#selecting only un correlated variables in the training set

train <- train[non_correlated_vars]
train2 <- train

library(ROSE)
#Performing oversampling for balancing the data
data.balanced <- ovun.sample(TARGET~., p=0.24,data=train, method="over")
train <- data.balanced$data
print(prop.table(table(train$TARGET)))
train$TARGET <- as.factor(train$TARGET) #Category target variable is read as Integer by default in R

#performing randomForest on oversampled data
library(randomForest)

mySampSize <- ceiling(table(train$TARGET))

#rf = randomForest(train$TARGET ~ .,data=train,ntree=50, replace = FALSE, do.trace = 5,strata = train$TARGET,sampsize=c(mySampSize[1],mySampSize[2]))
rf = randomForest(train$TARGET ~ .,data=train,ntree=50, replace = FALSE, do.trace = 5,strata = train$TARGET, mtry = 16)

#Calculating the Area under Curve for RandomForest
#library(AUC)
#rocc <- roc(predictor = as.numeric(predict(rf,data = train)),response = train$TARGET, plot = TRUE,)
#print(rocc)

library(pROC)
aucval <- auc(predictor = as.ordered(predict(rf,data = train)),response = train$TARGET,plot  = TRUE)
print(aucval)

#Performing logistic regression
library(SDMTools)
train$var38 <- log(train$var38)

mylogit <- glm(train$TARGET ~ ., data = train, family = "binomial")

#Cauculating AUC for logistic regression
logit_auc <- roc(predictor = as.ordered(predict(mylogit,type = "response",newdata = train)),response = train$TARGET, plot = TRUE)
print(logit_auc)

#Printing confusion matrix for logistic regression
confusion.matrix(train$TARGET,predict(mylogit,type = "response",newdata = train))

#Performing Gradient Boosting
library(xgboost)
library(Matrix)
train <- train2
train.target <- train$TARGET
train2 <- sparse.model.matrix(train$TARGET ~ ., data = train)
dtrain <- xgb.DMatrix(data=train2, label=train$TARGET)
watchlist <- list(train=dtrain)

paramlist <- list(  objective           = "binary:logistic", 
                    booster             = "gbtree",
                    eval_metric         = "auc",
                    eta                 = 0.0202048,
                    max_depth           = 5,
                    subsample           = 0.6815,
                    colsample_bytree    = 0.701
)

clf <- xgb.train(   params              = paramlist, 
                    data                = dtrain, 
                    nrounds             = 150, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

#Calculating AUC for Gradient Boosting
#xgb_auc <- roc(predictor = as.ordered(predict(clf,type = "response",newdata = train)),response = train$TARGET, plot = TRUE)
#print(xgb_auc)

xgb_rocc <- roc(predictor = as.numeric(predict(clf,newdata = dtrain)),response = train$TARGET, plot = TRUE)
print(xgb_rocc)

#Predicting outputs on the test data
test = read.csv("../input/test.csv")
testID <- test$ID       #saving Tes Ids for submission purpose
test$ID <- NULL
non_correlated_vars <- non_correlated_vars[! non_correlated_vars %in% "TARGET"]
test <- test[non_correlated_vars]
test = replace(test,NULL,NA)
test$var38 <- log(test$var38)
TARGET <- predict.glm(mylogit,type = "response", newdata = test)
TARGET <- predict(rf,type = "response",newdata = test)

test2 <- sparse.model.matrix(testID ~ ., data = test)
dtest <- xgb.DMatrix(data=test2, label=testID)
TARGET <- predict(clf,newdata = dtest)

ID <- testID
Output <- data.frame(ID,TARGET)
write.csv(Output,"Output.csv",row.names = FALSE)
