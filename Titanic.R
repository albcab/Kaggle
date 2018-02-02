# Titanic: Machine Learning from Disaster ---------------------------------

setwd("~/Data Science/Kaggle/Titanic")
file.exists(c("train.csv", "test.csv"))
library(Amelia)
library(psych)
library(tidyverse)
library(modelr)
library(magrittr)

col_types <- cols(Sex = col_factor(c("male", "female")),
                  Embarked = col_factor(c("C", "Q", "S")),
                  Pclass = col_factor(c("1", "2", "3")))
train <- read_csv("train.csv", col_types = col_types); train
test <- read_csv("test.csv", col_types = col_types); test
summary(train)
summary(test)

# Visualization -----------------------------------------------------------

train %>% ggplot() +
  geom_bar(aes(Survived)) +
  facet_wrap(Sex~Pclass)

train %>% ggplot() +
  geom_point(aes(y = Fare, x = Age, color = Survived))

train %>% group_by(Survived) %>% summarise(mean_fare = mean(Fare),
                                           count = n())
train %>% group_by(Sex) %>% summarise(pct_live = mean(Survived),
                                      count = n())

train %>% ggplot(aes(y = SibSp, x = Parch)) +
  geom_point() +
  geom_jitter() +
  facet_grid(~Survived)


# Feature Ing -------------------------------------------------------------

test %<>% mutate(Survived = NA)
train <- rbind(train, test)

library(stringr)

strings <- train %>% select(Name, Ticket, Cabin)
unique(strings$Cabin)
strings %$% sum(is.na(Cabin))

train %>% filter(Fare < 500) %>% ggplot() +
  geom_histogram(aes(Fare, fill = factor(Survived)))

count_cabins <- function(x) {
  a <- str_count(x, " ")
  if(is.na(a)) return(0)
  return(a+1)
}

train2 <- train %>% 
  separate(Name, c("Last", "Title", "First"), sep = "[,\\.] ", extra = "merge") %>%
  separate(Ticket, c("TktPre", "TktSuf"), sep = " ", extra = "merge", fill = "left") %>%
  mutate(Deck = str_extract(Cabin, "[a-zA-z]"),
         NumCabins = map_dbl(Cabin, count_cabins),
         NumCabins = parse_factor(NumCabins, levels = unique(NumCabins)),
         Farecut = factor(map_chr(Fare, cut, c(-Inf, 50, Inf))),
         Family = Parch + SibSp)

train2 %$% table(NumCabins, Survived)
train2 %>% group_by(NumCabins) %>% summarise(count = n(),
                                             pct = mean(Survived, na.rm = T))

train2 %>% group_by(Farecut) %>% summarise(count = n(),
                                             pct = mean(Survived, na.rm = T))

train2 %>% group_by(Title) %>% 
  # filter(PassengerId <= (1309-418)) %>%
  summarise(count = n(), 
            pct_sur = mean(Survived, na.rm = T)) %>% print(n=100)

library(forcats)

train2$Title %<>% parse_factor(levels = unique(train2$Title))
train2$Title2 <- fct_collapse(train2$Title, "Mr" = c("Don", "Capt", "Mr", "Rev"),
                               "Master" = c("Jonkheer", "Master"),
                               "Miss" = c("Miss", "Mlle"),
                               "Mrs" = c("Mme", "Mrs", "Ms", "Lady", "the Countess", "Dona"),
                               "Royal" = c("Col", "Dr", "Major", "Sir"))

# NAs ---------------------------------------------------------------------

train %>% filter(Fare < 1) %>% select(Fare, Age, Sex, Pclass, Survived)
# Supongamos que hubo gente que no pago el boleto, todos menos 1 mueren

## Age
ggplot(train2, aes(x= Age, y = Parch)) +
  geom_point() +
  geom_smooth()

train2 %>% filter(is.na(Age)) %>% group_by(Title, SibSp) %>% 
  summarise(count = n()) %>% print(n=100)

na_age <- lm(Age~ Title + SibSp, train2)
summary(na_age)
predict2 <- train2 %>% group_by(Title) %>% summarise(mean = mean(Age, na.rm = T))

train2$predict <- predict(na_age, train2)
train2 <- left_join(train2, predict2, by = "Title")

train2 %>% filter(!is.na(Age)) %>% select(Age, predict, mean) %>% 
  mutate(diff1 = abs(Age - predict),
         diff2 = abs(Age - mean)) %>% summarise(mean(diff1), mean(diff2))

for (i in seq_along(train2$Age)) {
  if (is.na(train2$Age[i])) {
    train2$Age[i] <- train2$predict[i]
  }
}

train2$Age[train2$Age < 0] <- 0

train2 %<>% select(-mean, -predict)

## Cabin

library(nnet)

train2 %>% group_by(TktSuf) %>% summarize(count = n()) %>% View()

train2 <- mutate(train2, TktSuf = str_extract(TktSuf, "[0-9]{2,}"))
train2$TktSuf <- parse_integer(train2$TktSuf)
train2$Deck <- parse_factor(train2$Deck, unique(train2$Deck)[-1])

na_deck <- multinom(Deck~ TktSuf, data = train2)
summary(na_deck)

deck_prediction <- predict(na_deck, train2)
for (i in seq_along(train2$Deck)) {
  if(is.na(train2$Deck[i])) {
    train2$Deck[i] <- deck_prediction[i]
  }
}

missmap(train2)
summary(train2)

# Modelling ---------------------------------------------------------------

train_model <- train2 %>% 
  filter(PassengerId <= 891) %>%
  select(-Last, -Title, -First, -TktPre, -TktSuf, -Cabin)
test_model <- train2 %>%
  filter(PassengerId > 891) %>%
  select(-Survived, -Last, -Title, -First, -TktPre, -TktSuf, -Cabin)
summary(train_model)
pairs.panels(train_model[,-1])

library(caret)

## Linear Model
lmodel <- lm(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck, 
             train_model)
summary(lmodel)

grid <- train_model %>%
  add_predictions(lmodel) %>%
  mutate(pred = round(pred)) %>%
  add_residuals(lmodel) %>%
  select(PassengerId, Survived, pred, resid)

confusionMatrix(grid$pred, grid$Survived)

ggplot(grid, aes(y = resid, x = PassengerId)) + 
  geom_point() + 
  geom_ref_line(0)

## Logistic Regression
logitmodel <- glm(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck, 
                  family = binomial("logit"), train_model)
summary(logitmodel)

gridlogit <- train_model %>%
  mutate(predi = predict(logitmodel, train_model, type = "response"),
         pred =  ifelse(predi <= 0.5, 0, 1)) %>%
  add_residuals(logitmodel) %>%
  select(PassengerId, Survived, pred, resid)
  
confusionMatrix(gridlogit$pred, gridlogit$Survived)

ggplot(gridlogit, aes(PassengerId, resid)) + 
  geom_point() + 
  geom_ref_line(0)

## Decision trees/Random Forest
train_model %<>% mutate(Survived = parse_factor(Survived, levels = unique(Survived)))
library(C50)
dtmodel <- C5.0(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck, train_model)
summary(dtmodel)

library(randomForest)
rfmodel <- randomForest(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck, 
                        train_model, na.action = na.omit, mtry = 3)
rfmodel
importance(rfmodel)

## Naive Bayes Classification
library(e1071)
#train_model %<>% mutate(Survived = parse_factor(Survived, levels = unique(Survived)))
nbmodel <- naiveBayes(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck,
                      train_model)

gridnb <- train_model %>%
  mutate(pred = predict(nbmodel, train_model)) %>%
  select(PassengerId, Survived, pred)

gridnb %$% confusionMatrix(pred, Survived)

## Neural Network
### Single layer
library(nnet)
nnmodel <- nnet(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2, 
                train_model, size = 10, na.action = na.omit)
summary(nnmodel)

gridnn <- train_model %>%
  add_predictions(nnmodel) %>%
  mutate(pred = ifelse(pred >= 0.5, 1, 0)) %>%
  select(PassengerId, Survived, pred)

gridnn %$% confusionMatrix(pred, Survived)

### Multiple Layers
#### Feed Forward Network
library(mxnet) #installed on CPU

nn_ready <- function(x) {
  if (is.numeric(x) | is.logical(x)) return(as.numeric(x))
  if (is.character(x)) stop("Cannot be character, only num/factor/logical")
  u <- unique(x)
  u <- u[!is.na(u)]
  if (length(u) <= 2) return(as.numeric(x) - 1)
  new <- vector("list", length = length(u)-1)
  for (i in seq_along(new)) {
    new[[i]] <- rep(0, length(x))
    new[[i]][x == u[i]] <- 1
    new[[i]] <- as.numeric(new[[i]])
  }
  new <- as.data.frame(new)
  names(new) <- NULL
  return(new)
}

train.nn <- as.data.frame(map(train_model, nn_ready))

train.x <- data.matrix(train.nn[, c(-1, -2, -7, -8, -10:-18)])
train.y <- train.nn[, 2]
mxmodel <- mx.mlp(train.x, train.y, hidden_node=c(10,10), out_node=2, activation="sigmoid",
                  num.round=50, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                  eval.metric=mx.metric.accuracy)

pred <- predict(mxmodel, train.x)
confusionMatrix(ifelse(pred[2,] >= 0.5, 1, 0), train.y)


# Model -------------------------------------------------------------------

fix_na <- function(x) {
  for(i in seq_along(x)) {
    if(is.na(x[i])) x[i] <- sample(na.omit(x), 1)
  }
  return(x)
}

train_model <- as_tibble(map(train_model, fix_na))

train_cv <- createDataPartition(train_model$Survived, p = 0.6, list = F)
trainer <- train_model[train_cv,]
cv <- train_model[-train_cv,]

## Logit model
train_logit <- glm(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck,
                   family = binomial("logit"), data = trainer)

cv.logitgrid <- cv %>%
  add_predictions(train_logit, var = "response") %>%
  mutate(response = ifelse(response >= 0.5, 1, 0)) %>%
  select(PassengerId, Survived, response)

cv.logitgrid %$% confusionMatrix(Survived, response)

## Random Forest
cv_rf <- list()
acc <- vector(length = 9)
for(i in seq(100, 2000, by = 100)) {
  cv_rf <- randomForest(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked,
                           trainer, mtry = 1, ntree = i)
  cv.rfgrid <- cv %>% 
    add_predictions(cv_rf) %>%
    select(PassengerId, Survived, pred)
  acc[i/100] <- cv.rfgrid %$% confusionMatrix(Survived, pred)$overall[1]
}
plot(acc)
lines(acc, col="salmon")

# Neural Network
cv_nn <- list()
pred_nn <- vector(length = 10)
for(i in 5:20) {
  cv_nn <- nnet(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked,
                trainer, size = i)
  cv.nngrid <- cv %>% 
    add_predictions(cv_nn) %>%
    mutate(pred = ifelse(pred >= 0.5, 1, 0)) %>%
    select(PassengerId, Survived, pred)
  pred_nn[i-4] <- cv.nngrid %$% confusionMatrix(Survived, pred)$overall[1]
}
plot(pred_nn)
lines(pred_nn, col = "salmon")

# Prediction --------------------------------------------------------------

prediction <- glm(Survived~ Pclass + Sex + Age + Family + Farecut + NumCabins + Title2 + Embarked + Deck,
                  family = binomial("logit"), data = train_model)

test_model <- as_tibble(map(test_model, fix_na))

pred <- test_model %>% 
  add_predictions(prediction) %>%
  mutate(Survived = ifelse(pred >= 0.5, 1, 0)) %>%
  select(PassengerId, Survived)

write_csv(pred, "titanicpredictions.csv")
