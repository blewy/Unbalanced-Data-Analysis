# See this code
# http://appliedpredictivemodeling.com/blog/2014/2/1/lw6har9oewknvus176q4o41alqw2ow

library(caret)
library(pROC)

set.seed(442)
training <- twoClassSim(n = 1000, intercept = -16)
testing <- twoClassSim(n = 1000, intercept = -16)

table(training$Class)


set.seed(949)
mod0 <- train(Class ~ ., data = training,
        method = "rf",
        metric = "ROC",
        tuneGrid = data.frame(mtry = 3),
        trControl = trainControl(method = "cv",
        classProbs = TRUE,
        summaryFunction = twoClassSummary))


getTrainPerf(mod0)

#How do we optimize this model? Normally we might look at the area under the ROC curve as a metric to choose our final values. In this case the ROC curve is independent of the probability threshold so we have to use something else. A common technique to evaluate a candidate threshold is see how close it is to the perfect model where sensitivity and specificity are one. Our code will use the distance between the current model's performance and the best possible performance and then have train minimize this distance when choosing it's parameters. Here is the code that we use to calculate this:

fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  ## This code will get use the area under the ROC curve and the
  ## sensitivity and specificity values using the current candidate
  ## value of the probability threshold.
  out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
  
  ## The best possible model has sensitivity of 1 and specifity of 1. 
  ## How far are we from that value?
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]), 
                   ncol = 2, 
                   byrow = TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  c(out, Dist = dist(coords)[1])
}


set.seed(949)
mod1 <- train(Class ~ ., data = training,
              ## 'modelInfo' is a list object found in the linked
              ## source code
              method = "rf",
              ## Minimize the distance to the perfect model
              metric = "Dist",
              maximize = FALSE,
              tuneLength = 20,
              trControl = trainControl(method = "cv",
                                       classProbs = TRUE,
                                       summaryFunction = fourStats))

mod1
