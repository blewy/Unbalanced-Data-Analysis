library(caret)
library(ggplot2)
library(grid)
library(gridExtra)


set.seed(2969)
imbal_train <- twoClassSim(1000, intercept = -10, linearVars = 0)

#table(imbal_train$Class)

# Set shape by cond
p<- ggplot(imbal_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Original Imbalance") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5)) 
p

set.seed(2969)
down_train <- downSample(x = imbal_train[, -ncol(imbal_train)],
                         y = imbal_train$Class)
table(down_train$Class)

# Set shape by cond
p1<- ggplot(down_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Down Sampled")+ coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))
p1


set.seed(9560)
up_train <- upSample(x = imbal_train[, -ncol(imbal_train)],
                     y = imbal_train$Class)
table(up_train$Class)

# Set shape by cond
p2<- ggplot(up_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Up Sampled")+ coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))

p2


library(DMwR)

set.seed(9560)
smote_train <- SMOTE(Class ~ ., data  = imbal_train)
table(smote_train$Class)


# Set shape by cond
p3<- ggplot(smote_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Smote Sample")+ coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))
p3

library(ROSE)

set.seed(9560)
rose_train <- ROSE(Class ~ ., data  = imbal_train)$data
table(rose_train$Class)

# Set shape by cond
p4<- ggplot(rose_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point() + theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Rose Sample")+ coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))
p4

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

multiplot( p1, p2, p3, p4, cols=2)
