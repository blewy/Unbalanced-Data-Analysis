library(xgboost)
library(Matrix)
library(caret)
library(pROC)

train = read.csv("../input/train.csv")
test = read.csv("../input/test.csv")

### Removing IDs
train$ID = NULL
test.id = test$ID
test$ID = NULL

### Extracting TARGET
train.y = train$TARGET
train$TARGET = NULL

### 0 count per line
count0 = function(x) {
  return( sum(x == 0) )
}
train$n0 = apply(train, 1, FUN=count0)
test$n0 = apply(test, 1, FUN=count0)

### Removing constant features
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. Removing.\n")
    train[[f]] = NULL
    test[[f]] = NULL
  }
}

### Removing identical features
features_pair = combn(names(train), 2, simplify = F)
toRemove = c()
for(pair in features_pair) {
  f1 = pair[1]
  f2 = pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove = c(toRemove, f2)
    }
  }
}

feature.names = setdiff(names(train), toRemove)

train$var38 = log(train$var38)
test$var38 = log(test$var38)

train = train[, feature.names]
test = test[, feature.names]
tc = test

### Limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
for(f in colnames(train)){
  lim = min(train[, f])
  test[test[, f] < lim,f] = lim
  
  lim = max(train[, f])
  test[test[, f] > lim,f] = lim  
}

### Remove linear combos
combo = findLinearCombos(train)
train = train[, -combo$remove]

train$TARGET <- train.y

### 20% for random grid search validation
index = sample(1:nrow(train), nrow(train)*0.8, replace = F)
train.tune = train[index, ]
train.valid = train[-index, ]
train.tune.y = train.tune$TARGET
train.valid.y = train.valid$TARGET

train.tune = sparse.model.matrix(TARGET ~ ., data = train.tune)
train.valid = sparse.model.matrix(TARGET ~ ., data = train.valid)

dtune = xgb.DMatrix(data = train.tune, label = train.tune.y)

best.param = list()
best.seed = 0
best.auc = 0
best.auc.index = 0

### XGBoost random grid search
for (iter in 1:10){
  param = list(objective = 'binary:logistic',
               eval_metric = 'auc',
               max_depth = sample(4:8, 1),
               eta = round(runif(1, 0.01, 0.03), 4),
               gamma = round(runif(1, 0.0, 0.2), 4),
               subsample = round(runif(1, 0.6, 0.9), 4),
               colsample_bytree = round(runif(1, 0.5, 0.8), 4),
               min_child_weight = sample(1:40, 1),
               max_delta_step = sample(1:10, 1)
  )
  seed.number = sample.int(1000, 1)[[1]]
  set.seed(seed.number)
  cat("Iteration", iter, "for random grid search. \n")
  cv = xgb.cv(params = param, data = dtune, nfold = 5, nrounds = 1000, verbose = F, early.stop.round = 10, maximize = T)
  max.auc = max(cv[, test.auc.mean])
  max.auc.index = which.max(cv[, test.auc.mean])
  
  if (max.auc > best.auc){
    best.auc = max.auc
    best.auc.index = max.auc.index
    best.seed = seed.number
    best.param = param
  }
  cat("", sep = "\n\n")
}

set.seed(best.seed)
xgb.valid.fit = xgb.train(data = dtune, params = best.param, nrounds = best.auc.index, verbose = T, watchlist = tunewatch, maximize = F)

valid.pred = predict(xgb.valid.fit, train.valid)
AUC = function(actual, predicted)
{
  auc = auc(as.numeric(actual),as.numeric(predicted))
  auc 
}
### Validation AUC
AUC(train.valid.y, valid.pred) 

train = sparse.model.matrix(TARGET ~ ., data = train)
dtrain = xgb.DMatrix(data = train, label = train.y)
watchlist = list(train = dtrain)

xgb.fit = xgb.train(data = dtrain, params = best.param, nrounds = best.auc.index, verbose = T, watchlist = watchlist, maximize = F)

test$TARGET = -1
test = sparse.model.matrix(TARGET ~ ., data = test)

preds = predict(xgb.fit, test)

### Manual prediction adjustments from ZFTurbo
nv = tc['num_var33']+tc['saldo_medio_var33_ult3']+tc['saldo_medio_var44_hace2']+tc['saldo_medio_var44_hace3']+tc['saldo_medio_var33_ult1']+tc['saldo_medio_var44_ult1']

preds[nv > 0] = 0
preds[tc['var15'] < 23] = 0
preds[tc['saldo_medio_var5_hace2'] > 160000] = 0
preds[tc['saldo_var33'] > 0] = 0
preds[tc['var38'] > 3988596] = 0
preds[tc['var21'] > 7500] = 0
preds[tc['num_var30'] > 9] = 0
preds[tc['num_var13_0'] > 6] = 0
preds[tc['num_var33_0'] > 0] = 0
preds[tc['imp_ent_var16_ult1'] > 51003] = 0
preds[tc['imp_op_var39_comer_ult3'] > 13184] = 0
preds[tc['saldo_medio_var5_ult3'] > 108251] = 0
preds[tc['num_var37_0'] > 45] = 0
preds[tc['saldo_var5'] > 137615] = 0
preds[tc['saldo_var8'] > 60099] = 0
preds[(tc['var15']+tc['num_var45_hace3']+tc['num_var45_ult3']+tc['var36']) <= 24] = 0
preds[tc['saldo_var14'] > 19053.78] = 0
preds[tc['saldo_var17'] > 288188.97] = 0
preds[tc['saldo_var26'] > 10381.29] = 0
preds[tc['num_var13_largo_0'] > 3] = 0
preds[tc['imp_op_var40_comer_ult1'] > 3639.87] = 0

submission = data.frame(ID = test.id, TARGET = preds)
cat("saving the submission file\n")
write.csv(submission, "submission_rand_xgbgrid.csv", row.names = F)
cat("saving best params\n")
write.csv(best.param,"best_param.csv", row.names = F)
