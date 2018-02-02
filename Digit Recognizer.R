###########################################################################
## Digit Recognizer ######################################### MNIST DATA ##
###########################################################################

library(tidyverse)
library(caret)
library(magrittr)

setwd("~/Data Science/Kaggle/Digit Recognizer")
file.exists("train.csv", "test.csv")
train <- read_csv("train.csv")
test <- read_csv("test.csv")
test$label <- NA
duo <- rbind(train, test)
table(as.factor(train$label))

# Visualization -----------------------------------------------------------

ggplot(train, aes(x = as.factor(label), fill = label)) +
  geom_bar(color = "black") + 
  scale_fill_gradient(low = "red", high = "green", guide = F) +
  labs(x = "Numbers in data", title = "Train Data")

img <- matrix(t(duo[1,-1]), ncol = 28, nrow = 28)
for(i in 1:28) {
  img[i,] <- rev(img[i,])
}
image(img, col = grey.colors(100), axes = F)

first50 <- t(train[1:50, -1])
imgs <- map(1:50, function(x) matrix(first50[,x], ncol = 28, nrow = 28, byrow = F))
par(mfrow = c(5, 10), mar = c(.1, .1, .1, .1))
for(i in seq_along(imgs)) {
  for(j in 1:28) {
    imgs[[i]][j,] <- rev(imgs[[i]][j,])
  }
  image(imgs[[i]], col = grey.colors(100), axes = F)
  #readline("Presione Enter para continuar:")
}

# PCA ---------------------------------------------------------------------

nzv <- nearZeroVar(duo, freqCut = 10000/1, uniqueCut = 1/7, saveMetrics = T)
sum(nzv$zeroVar)
sum(nzv$nzv)
duo_nzv <- duo[,!nzv$nzv]

ft_scaling <- function(x) {
  if (all(unique(x) == 0)) return(x)
  return(as.vector(scale(x)))
}

duo_pca <- as_tibble(map(duo_nzv[,-1], ft_scaling))

sigma <- cov(duo_pca)
sing_val <- svd(sigma)
u <- sing_val$u
s <- sing_val$d

PCA <- 1:570
var_explained <- as_tibble(PCA)
var_explained %<>% mutate(var = map_dbl(PCA, function(x) return(sum(s[1:x]/sum(s)))))
ggplot(var_explained, aes(x = value, y = var)) +
  geom_point() +
  geom_line()

u350 <- u[, 1:100]
duo_pca350 <- as.matrix(duo_pca) %*% u350
duo2 <- as_tibble(cbind(as.matrix(duo$label), duo_pca350))
names(duo2)[1] <- "label"

## Visualize with PCA (2D, 3D)
u2 <- u[, 1:2]
dim(u2)
duo_pca2d <- as.matrix(duo_pca) %*% u2
dim(duo_pca2d)
duo_pca2d <- as_tibble(cbind(as.matrix(duo$label), duo_pca2d))
ggplot(duo_pca2d[1:42000,]) +
  geom_point(aes(x = V2, y = V3, color = factor(V1))) +
  scale_color_brewer(palette = "Spectral")

## Alternative way of doing PCA

prin_comp <- prcomp(sigma, center = F)
pca2 <- as_tibble(list(num = 1:length(prin_comp$sdev)))
pca2 %<>% mutate(var = prin_comp$sdev^2/sum(prin_comp$sdev^2),
                 cum = cumsum(var))
ggplot(pca2[1:100,], aes(x = num, y = cum)) +
  geom_point() +
  geom_line()

# Modeling ----------------------------------------------------------------

train2 <- duo2[1:42000,]
train2$label %<>% factor(levels = sort(unique(train2$label)))
test2 <- duo2[42001:70000, -1]

## SVM

control <- trainControl(method = "cv",
                        number = 5)
tune <- data.frame(#sigma = 0.01104614,
                   sigma = 0.01,
                   C = 3.5)
svm_model <- train(label~., 
                   train2,
                   method = "svmRadial",
                   #tuneLength = 9,
                   trControl = control,
                   tuneGrid = tune)
names(svm_model); svm_model
predictions <- as_tibble(list(ImageId = 1:nrow(test)))
predictions %<>% mutate(Label = predict(svm_model$finalModel, test2, type = "response"))
write_csv(predictions, "svmpred.csv")

## Neural Networks
# First, create function for cost, regardless of total/hidden layers and nodes
# then compute gradient
# finally, gradient check

sigmoid <- function(x) {
  return(1/(1+exp(-x)))
}

forward_prop <- function(x, theta, layers = length(theta)+1) { #forward propagation function
  fwd <- list(matrix(nrow = nrow(x), ncol = ncol(x)))
  for (l in 2:layers) {
    if (l < layers) fwd[[l]] <- matrix(nrow = nrow(x), ncol = ncol(theta[[l-1]])+1)
    else fwd[[l]] <- matrix(nrow = nrow(x), ncol = ncol(theta[[l-1]]))
  }  
  for (i in 1:nrow(x)) {
    a <- x[i,]
    if (!is.matrix(x)) a <- as.matrix(a)
    fwd[[1]][i,] <- a 
    for (j in 2:layers) {
      z <- a %*% theta[[j-1]]
      a <- c(1, sigmoid(z))
      if (j < layers) fwd[[j]][i,] <- a 
    }
    h_x <- a[-1]
    fwd[[layers]][i,] <- h_x
  }
  return(fwd)
}

pre_proc <- function(X, y, hiddenl = 1, nodes = 5, E = 2) { #Initialize random thetas, modify x and y
  m <- nrow(X)
  y <- as.integer(unlist(y))
  x <- cbind(rep(1, m), X)
  k <- length(unique(y))
  class <- sort(unique(y))
  if (k < 2) stop("y must be a categorical variable w/ at least 2 classes")
  if (k > 2) {
    Y <- matrix(rep(0, k*m), ncol = k)
    for (i in 1:m) {
      Y[i, y[i] == class] <- 1
    }
  } else {
    Y <- rep(0, m)
    for (i in 1:m) {
      if (y[i] == class[2]) Y[i] <- 1
    }
  }
  theta <- list()
  var <- ncol(x)
  for (i in 1:(hiddenl+1)) {
    if (i == 1) {
      theta[[i]] <- matrix(runif(var*nodes, -E, E), ncol = nodes, nrow = var)
      next
    }
    if (i != (hiddenl+1)) {
      theta[[i]] <- matrix(runif((nodes+1)*nodes, -E, E), ncol = nodes)
    } else {
      ifelse(k == 2, theta[[i]] <- matrix(runif(nodes+1, -E, E), ncol = 1),
                     theta[[i]] <- matrix(runif((nodes+1)*k, -E, E), ncol = k))
    }
  }
  processed <- list(x = x, y = Y, theta = theta, theta_unrolled = unlist(theta))
  return(processed)
}

cost_nn <- function(x, y, theta, lambda, l = length(theta)+1) { # Cost function
  J <- 0
  m <- nrow(x)
  h_x <- forward_prop(x, theta)[[l]]
  for (i in 1:(l-1)) {
    theta[[i]] <- theta[[i]][-1,]
  }
  J <- -sum(sum(y*log(h_x)+(1-y)*log(1-h_x)))/m + (lambda/(2*m))*sum(unlist(theta)^2)
  return(J)
}

sigma_grad <- function(x) {
  return(x * (1-x))
}

back_prop <- function(x, y, theta, lambda) { #back propagation function
  m <- nrow(x)
  big_delta <- list()
  fwd <- forward_prop(x, theta)
  layers <- length(fwd)-1
  delta <- fwd[[layers+1]] - y
  big_delta[[layers]] <- t(fwd[[layers]]) %*% delta
  for (i in layers:2) {
    delta <- delta %*% t(theta[[i]]) * sigma_grad(fwd[[i]])
    delta <- delta[,-1]
    big_delta[[i-1]] <- t(fwd[[i-1]]) %*% delta
  }
  graD <- big_delta
  for (i in 1:layers) {
    graD[[i]][1,] <- big_delta[[i]][1,]/m
    graD[[i]][-1,] <- (big_delta[[i]][-1,] + lambda*theta[[i]][-1,])/m
  }
  return(graD)
}

## Gradient checking
grad_check <- function(x, y, theta, lambda, epsilon = 1e-4) {
  gcheck <- theta
  for (i in 1:length(theta)) {
    for (j in 1:length(theta[[i]])) {
      thetaplus <- theta
      thetaminus <- theta
      thetaplus[[i]][j] <- theta[[i]][j] + epsilon
      thetaminus[[i]][j] <- theta[[i]][j] - epsilon
      gcheck[[i]][j] <- (cost_nn(x, y, thetaplus, lambda) - cost_nn(x, y, thetaminus, lambda))/(2*epsilon)
    }
  }
  return(gcheck)
}

## We need to set everything up to feed into optimization algorithm
## thetas must be in a vector
## functions must only need theta to compute (functions specifically for dataset)

nn_iris <- function(theta) {
  theta <- list(hidden = matrix(theta[1:25], ncol = 5),
                result = matrix(theta[26:43], ncol = 3))
  x <- Iris$x
  y <- Iris$y
  return(cost_nn(x, y, theta, 0))
}

nngrad_iris <- function(theta) {
  theta <- list(hidden = matrix(theta[1:25], ncol = 5),
                result = matrix(theta[26:43], ncol = 3))
  x <- Iris$x
  y <- Iris$y
  return(unlist(back_prop(x, y, theta, 0)))
}

library(nnet)

Iris <- pre_proc(iris[,-5], iris[,5])

nnet_iris <- optim(Iris$theta_unrolled, nn_iris, nngrad_iris, method = "BFGS")
theta_iris <- list(hidden = matrix(nnet_iris$par[1:25], ncol = 5), 
                   result = matrix(nnet_iris$par[26:43], ncol = 3))
pred <- forward_prop(Iris$x, theta_iris)[[3]]
predictions <- matrix(rep(0, prod(dim(pred))), ncol = ncol(pred))
pred <- apply(pred, 1, which.max)
for (i in 1:nrow(predictions)) {
  predictions[i, pred[i]] <- 1
}
confusionMatrix(Iris$y, predictions)
# vs.
base <- nnet(Iris$x, Iris$y, size = 5)
confusionMatrix(Iris$y, round(predict(base)))

## NOW FOR THE REAL DEAL  

digits <- pre_proc(train2[,-1], train2[,1], 2, 5)

nn_digits <- function(theta) {
  theta <- list(hidden1 = matrix(theta[1:505], ncol = 5),
                hidden2 = matrix(theta[506:535], ncol = 5),
                result = matrix(theta[536:595], ncol = 10))
  x <- digits$x
  y <- digits$y
  return(cost_nn(x, y, theta, 2))
}

nngrad_digits <- function(theta) {
  theta <- list(hidden1 = matrix(theta[1:505], ncol = 5),
                hidden2 = matrix(theta[506:535], ncol = 5),
                result = matrix(theta[536:595], ncol = 10))
  x <- digits$x
  y <- digits$y
  return(unlist(back_prop(x, y, theta, 2)))
}

nnet_digits <- optim(digits$theta_unrolled, nn_digits, nngrad_digits, method = "BFGS")
theta_digits <- list(hidden1 = matrix(nnet_digits$par[1:505], ncol = 5),
                     hidden2 = matrix(nnet_digits$par[506:535], ncol = 5),
                     result = matrix(nnet_digits$par[536:595], ncol = 10))
pred <- forward_prop(digits$x, theta_digits)[[4]]
pred <- apply(pred, 1, which.max)
pred <- pred - 1
confusionMatrix(unlist(train2[,1]), pred)

digits_test <- pre_proc(test2, train2[,1], 2, 5)
pred_test <- forward_prop(digits_test$x, theta_digits)[[4]]
pred_test <- apply(pred_test, 1, which.max)
pred_test <- pred_test - 1
pred_test <- as_tibble(list(ImageId = 1:length(pred_test),
                            Label = pred_test))
write_csv(pred_test, "selfnnpred.csv")

## Neural Network using MXNetR

library(mxnet)

# Using the multi layer perceptron
train.y <- as.numeric(iris[,5]) - 1
train.x <- t(iris[, 1:4])
iris_mxnet <- mx.mlp(train.x, train.y, hidden_node = 10, out_node = 3, 
                     activation = "relu", 
                     out_activation = "softmax", 
                     num.round = 100,
                     learning.rate = 0.1, eval.metric = mx.metric.accuracy)
pred_mxnet <- predict(iris_mxnet, train.x)
max.col(t(pred_mxnet)) - 1

# Using the Symbol system
data <- mx.symbol.Variable("data")
z1 <- mx.symbol.FullyConnected(data, name = "z1", num_hidden = 10)
a1 <- mx.symbol.Activation(z1, name = "a1", act_type = "relu")
z2 <- mx.symbol.FullyConnected(a1, name = "z2", num_hidden = 3)
hx <- mx.symbol.SoftmaxOutput(z2, name = "hx")
iris_mxnet2 <-  mx.model.FeedForward.create(hx, X = train.x, y = train.y,
                                            ctx = mx.cpu(),
                                            num.round = 100,
                                            learning.rate = 0.1,
                                            eval.metric = mx.metric.accuracy)
pred_mxnet2 <- predict(iris_mxnet2, train.x)
max.col(t(pred_mxnet2)) - 1

## NOW FOR THE REAL DEAL
test <- read_csv("test.csv")

train.x <- t(train[,-1]/255)
train.y <- unlist(train[,1])
test.x <- t(test/255)

data <- mx.symbol.Variable("data")
z1 <- mx.symbol.FullyConnected(data, num_hidden = 128)
a1 <- mx.symbol.Activation(z1, act_type = "relu")
z2 <- mx.symbol.FullyConnected(a1, num_hidden = 64)
a2 <- mx.symbol.Activation(z2, act_type = "relu")
z3 <- mx.symbol.FullyConnected(a2, num_hidden = 10)
hx <- mx.symbol.SoftmaxOutput(z3)
digits_mxnet <- mx.model.FeedForward.create(hx, X = train.x, y = train.y,
                                            array.layout = "colmajor",
                                            eval.metric = mx.metric.accuracy,
                                            ctx = mx.cpu(),
                                            array.batch.size = 100,
                                            momentum = 0.9,
                                            num.round = 10,
                                            learning.rate = 0.07,
                                            initializer = mx.init.uniform(0.07)) 
mxnet_pred <- max.col(t(predict(digits_mxnet, test.x, array.layout = "colmajor"))) - 1
mxnet_digits <- as_tibble(list(ImageId = 1:nrow(test), Label = mxnet_pred))
write_csv(mxnet_digits, "mxnet_nn.csv")
