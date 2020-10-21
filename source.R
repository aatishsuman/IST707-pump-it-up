library(ggplot2)
library(googleVis)
library(randomForest)
library(caret)
library(dplyr)
library(mlr)
library(xgboost)
library(parallel)
library(parallelMap)
library(e1071)
library(C50)

## Dataset URL - https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

setwd("C:/Data/study/IST-707/Project")

train_values <- read.csv("train.csv")
train_labels <- read.csv("trainLabels.csv")
test <- read.csv("test.csv")
train <- merge(train_labels, train_values)


#### Exploratory Analysis ####

table(train$status_group) 
prop.table(table(train$status_group))

table(train$payment, train$status_group)
prop.table(table(train$payment, train$status_group), margin = 1)

table(train$quantity, train$status_group)
prop.table(table(train$quantity, train$status_group), margin = 1)

qplot(funder, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top") + 
  theme(axis.text.x=element_text(angle = -20, hjust = 0))
funderCount = train %>% group_by(funder) %>% summarise(count_sales = n())
funderCount[funderCount$count_sales > 500,]

qplot(basin, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(region, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

length(unique(train$district_code))

qplot(public_meeting, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(permit, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(extraction_type, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(extraction_type_group, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(extraction_type_class, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(management, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(management_group, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(payment, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(payment_type, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(quantity, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(quantity_group, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(source, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(source_type, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(source_class, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top")

qplot(waterpoint_type, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top") +
  theme(axis.text.x=element_text(angle = -20, hjust = 0))

qplot(waterpoint_type_group, data=train, geom="bar", fill=status_group) + 
  theme(legend.position = "top") +
  theme(axis.text.x=element_text(angle = -20, hjust = 0))

ggplot(subset(train, amount_tsh > 100000), aes(x = amount_tsh)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, gps_height > 0), aes(x = gps_height)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, longitude > 0), aes(x = longitude)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, latitude > -100000), aes(x = latitude)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, num_private > -100000), aes(x = num_private)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, population > 0), aes(x = population)) +
  geom_histogram(binwidth = 100)

ggplot(train, aes(x = construction_year)) + 
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

ggplot(subset(train, construction_year > 0), aes(x = construction_year)) +
  geom_histogram(bins = 20) + 
  facet_grid( ~ status_group)

# Create scatter plot: latitude vs longitude with color as status_group
ggplot(subset(train, latitude < 0 & longitude > 0),
       aes(x = latitude, y = longitude, color = status_group)) + 
  geom_point(shape = 1) + 
  theme(legend.position = "top")

# Create a column 'latlong' to input into gvisGeoChart
train$latlong <- paste(round(train$latitude, 2), round(train$longitude, 2), sep=":")
train$Size <- 1

# Use gvisGeoChart to create an interactive map with the first 1000 well locations
wells_map <- gvisGeoChart(train[1:1000,], locationvar = "latlong", 
                          colorvar = "status_group", sizevar = "Size", 
                          options = list(region = "TZ"))
plot(wells_map)


#### Feature addition/selection ####

# Make installer lowercase, take first 3 letters as a sub string
train$install_3 <- substr(tolower(train$installer),1,3)
train$install_3[train$install_3 %in% c(" ", "", "0", "_", "-")] <- "other"

# Take the top 15 substrings from above by occurance frequency
install_top_15 <- names(summary(as.factor(train$install_3)))[1:15]
train$install_3[!(train$install_3 %in% install_top_15)] <- "other"
train$install_3 <- as.factor(train$install_3)

# Table of the install_3 variable vs the status of the pumps
table(train$install_3, train$status_group)

# As row-wise proportions, install_3 vs status_group
prop.table(table(train$install_3, train$status_group), margin = 1)

# Create install_3 for the test set using same top 15 from above
test$install_3 <- substr(tolower(test$installer),1,3)
test$install_3[test$install_3 %in% c(" ", "", "0", "_", "-")] <- "other"
test$install_3[!(test$install_3 %in% install_top_15)] <- "other"
test$install_3 <- as.factor(test$install_3)

# Make funder lowercase
train$funder_20 <- substr(tolower(train$funder),1,3)
train$funder_20[train$funder_20 %in% c(" ", "", "0", "_", "-")] <- "other"

# Take the top 20 substrings from above by occurance frequency
funder_top_20 <- names(summary(as.factor(train$funder_20)))[1:20]
train$funder_20[!(train$funder_20 %in% funder_top_20)] <- "other"
train$funder_20 <- as.factor(train$funder_20)

# Create funder_20 for the test set using same top 20 from above
test$funder_20 <- substr(tolower(test$funder),1,3)
test$funder_20[test$funder_20 %in% c(" ", "", "0", "_", "-")] <- "other"
test$funder_20[!(test$funder_20 %in% funder_top_20)] <- "other"
test$funder_20 <- as.factor(test$funder_20)


#### MNB ####
train_binned <- train %>%
  mutate(gps_height = cut(gps_height, breaks = c(min(gps_height), 500, 1500, max(gps_height)), labels = c("<500", "500-1500", ">1500"), include.lowest = TRUE),
         longitude = cut(longitude, breaks = c(min(longitude), 33, 35, 37, max(longitude)), labels = c("<33", "33", "35-37", ">37"), include.lowest = TRUE),
         latitude = cut(latitude, breaks = c(min(latitude), -10, -6, -3, max(latitude)), labels = c("<-10", "-10 - -6", "-6 - -3", ">-3"), include.lowest = TRUE),
         population = cut(population, breaks = c(min(population), 100, 1000, 10000, max(population)), labels = c("<100", "100-1000", "1000-10000", ">10000"), include.lowest = TRUE),
         construction_year = cut(construction_year, breaks = c(min(construction_year), 1960, 1980, 2000, 2010, max(construction_year)), labels = c("unknown", "1960-1980", "1980-2000", "2000-2010", ">2010"), include.lowest = TRUE)) %>%
  apply(MARGIN = 2, FUN = as.character) %>%
  data.frame()
set.seed(42)
inTraining <- createDataPartition(train_binned$status_group, p = 0.9, list = FALSE)
trainData <- train_binned[inTraining, colnames(train_binned) != "status_group"]
testData <- train_binned[-inTraining, colnames(train_binned) != "status_group"]
training_labels <- train_binned[inTraining, c("status_group")]
test_labels <- train_binned[-inTraining, c("status_group")]

# Training
model_nb <- naiveBayes(trainData, training_labels, laplace = 1)

# Testing
train_pred_nb <- predict(model_nb, trainData)
test_pred_nb <- predict(model_nb, testData)

# Confusion Matrix
confusionMatrix(train_pred_nb, training_labels)
confusionMatrix(test_pred_nb, test_labels)


#### Decision trees ####
ctrl <- caret::trainControl(method = "repeatedcv",
                            number = 3,
                            selectionFunction = "oneSE",
                            repeats = 5,
                            savePredictions = T)

grid <- expand.grid(model = "tree",
                    trials = c(1,5,10,15,20),
                    winnow = F)

model_dt <- caret::train(as.factor(status_group) ~ funder_20 + gps_height + install_3 + 
                           longitude + latitude + population + public_meeting + 
                           scheme_management + permit + construction_year + extraction_type_class + 
                           payment + quality_group + 
                           quantity + source_type + waterpoint_type_group + region,
                         data = train,
                         method = "C5.0",
                         metric = "Kappa",
                         trControl = ctrl,
                         tuneGrid = grid)
model_dt

# Training
train_dt <- train[, c("status_group","funder_20","gps_height","install_3",
                      "longitude","latitude","region","population","public_meeting",
                      "scheme_management","permit","construction_year","extraction_type_class",
                      "payment","quality_group",
                      "quantity","source_type","waterpoint_type_group")]
levels(train_dt$scheme_management)[1] = "unknown"
levels(train_dt$permit)[1] = "unknown"
levels(train_dt$public_meeting)[1] = "unknown"

set.seed(42)
inTraining <- createDataPartition(train_dt$status_group, p = 0.9, list = FALSE)
trainData <- train_dt[inTraining,]
testData <- train_dt[-inTraining,]
training_labels <- train_dt[inTraining, c("status_group")]
test_labels <- train_dt[-inTraining, c("status_group")]

best_model_dt <- trainData %>% 
  select(-c(status_group)) %>% 
  C5.0(training_labels, 
       trials = 20)

# Testing
best_model_dt_pred_train <- best_model_dt %>% 
  predict(trainData)
best_model_dt_pred_test <- best_model_dt %>% 
  predict(testData)

# Confusion Matrix
confusionMatrix(best_model_dt_pred_train, training_labels)
confusionMatrix(best_model_dt_pred_test, test_labels)


#### Random Forest ####

## default parameters
set.seed(42)
inTraining = createDataPartition(train$status_group, p = 0.9, list = FALSE)
trainData = train[inTraining,]
testData = train[-inTraining,]
model_forest <- randomForest(as.factor(status_group) ~ funder_20 + gps_height + install_3 + 
                               longitude + latitude + population + public_meeting + 
                               scheme_management + permit + construction_year + extraction_type_class + 
                               payment + quality_group + 
                               quantity + source_type + waterpoint_type_group + region,
                             data = trainData, importance = TRUE,
                             ntree = 50, nodesize = 2)
pred_forest_test <- predict(model_forest, testData)
confusionMatrix(pred_forest_test, testData$status_group)
varImpPlot(model_forest)

## tuning
modelList <- list()
for (nodesize in c(2:5)) {
  for (mtry in c(1:4)*2) {
    set.seed(42)
    inTraining = createDataPartition(train_scaled$status_group, p = 0.9, list = FALSE)
    trainData = train_scaled[inTraining,]
    testData = train_scaled[-inTraining,]
    model <- randomForest(as.factor(status_group) ~ funder_20 + gps_height + install_3 +
                            longitude + latitude + population + public_meeting +
                            scheme_management + permit + construction_year + extraction_type_class +
                            payment + quality_group +
                            quantity + source_type + waterpoint_type_group + region,
                          data = trainData, importance = TRUE,
                          mtry = mtry, ntree = 200, nodesize = nodesize)
    pred_forest_test <- predict(model, testData)
    conf_mat <- table(pred_forest_test, testData$status_group)
    accuracy <- sum(diag(conf_mat))/sum(conf_mat)
    key <- paste(c("nodesize","mtry"), c(nodesize, mtry), sep = "=", collapse = ", ")
    modelList[[key]] <- accuracy
  }
}

modelListDf <- as.data.frame(modelList)
modelListDf[which.max(as.data.frame(modelList))]

## best model
set.seed(42)
best_model_forest <- randomForest(as.factor(status_group) ~ funder_20 + gps_height + install_3 + 
                                    longitude + latitude + population + public_meeting + 
                                    scheme_management + permit + construction_year + extraction_type_class + 
                                    payment + quality_group + 
                                    quantity + source_type + waterpoint_type_group + region,
                                  data = train, importance = TRUE,
                                  mtry = 4, ntree = 200, nodesize = 4)
pred_forest_train <- predict(best_model_forest, train)
confusionMatrix(pred_forest_train, train$status_group)

## submission
pred_forest_test <- predict(best_model_forest, test)
submission <- data.frame(test$id)
submission$status_group <- pred_forest_test
names(submission)[1] <- "id"
write.csv(submission, "submission.csv", row.names = FALSE, quote = FALSE)


#### XGBoost ####
train_xgb <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                          data = train[, c("funder_20","gps_height","install_3",
                                           "longitude","latitude","region","population","public_meeting",
                                           "scheme_management","permit","construction_year","extraction_type_class",
                                           "payment","quality_group",
                                           "quantity","source_type","waterpoint_type_group")])

test_xgb <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                         data = test[, c("funder_20","gps_height","install_3",
                                         "longitude","latitude","region","population","public_meeting",
                                         "scheme_management","permit","construction_year","extraction_type_class",
                                         "payment","quality_group",
                                         "quantity","source_type","waterpoint_type_group")])

train_labels_xgb <- as.numeric(train$status_group)-1

dtrain <- xgb.DMatrix(data = train_xgb,label = train_labels_xgb)

# default parameters
params <- list(booster = "gbtree", objective = "multi:softmax", num_class=3, eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, early_stopping_rounds = 20, maximize = F)
plot(xgbcv$evaluation_log)

# eta and nrounds
set.seed(42)
inTraining = createDataPartition(train$status_group, p = 0.9, list = FALSE)
trainData = train[inTraining,]
testData = train[-inTraining,]

trainData_labels <- as.numeric(trainData$status_group)-1
testData_labels <- as.numeric(testData$status_group)-1

trainData <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                          data = trainData[, c("funder_20","gps_height","install_3",
                                               "longitude","latitude","region","population","public_meeting",
                                               "scheme_management","permit","construction_year","extraction_type_class",
                                               "payment","quality_group",
                                               "quantity","source_type","waterpoint_type_group","region")])

testData <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                         data = testData[, c("funder_20","gps_height","install_3",
                                             "longitude","latitude","region","population","public_meeting",
                                             "scheme_management","permit","construction_year","extraction_type_class",
                                             "payment","quality_group",
                                             "quantity","source_type","waterpoint_type_group","region")])

trainData <- xgb.DMatrix(data = trainData, label = trainData_labels)
testData <- xgb.DMatrix(data = testData, label = testData_labels)

params <- list(booster = "gbtree", objective = "multi:softmax", num_class=3, eta=0.1, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgb1 <- xgb.train (params = params, data = trainData, nrounds = 150, watchlist = list(val=testData,train=trainData), print_every_n = 50, early_stopping_rounds = 20, maximize = F , eval_metric = "merror")

xgbpred_train <- predict (xgb1, trainData)
confusionMatrix(factor(xgbpred_train), factor(trainData_labels)) # train accuracy

xgbpred_test <- predict (xgb1, testData)
confusionMatrix(factor(xgbpred_test), factor(testData_labels)) # test accuracy

# rest of the parameters
data <- train[, c("status_group","funder_20","gps_height","install_3",
                       "longitude","latitude","region","population","public_meeting",
                       "scheme_management","permit","construction_year","extraction_type_class",
                       "payment","quality_group",
                       "quantity","source_type","waterpoint_type_group")]
set.seed(42)

# create tasks and do one-hot encoding
traintask <- makeClassifTask (data = data, target = "status_group")
traintask <- createDummyFeatures (obj = traintask) 

# create learner and set parameter space and resampling strategy
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(booster="gbtree", objective="multi:softmax", eval_metric="merror", nrounds=150L, eta=0.1, gamma=2, max_depth=20, min_child_weight=1)
params <- makeParamSet(makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
rdesc <- makeResampleDesc("CV",stratify = T,iters=3L)

# search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

# set parallel backend
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

tune_params <- c(mytune$x, booster="gbtree", objective = "multi:softmax", num_class=3, eta=0.1, gamma=2, max_depth=20, min_child_weight=1)
set.seed(42)
inTraining = createDataPartition(data$status_group, p = 0.9, list = FALSE)
trainData = data[inTraining,]
testData = data[-inTraining,]

trainData_labels <- as.numeric(trainData$status_group)-1
testData_labels <- as.numeric(testData$status_group)-1

trainData <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, data = trainData %>% select(-c("status_group")))

testData <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                         data = testData %>% select(-c("status_group")))

trainData <- xgb.DMatrix(data = trainData, label = trainData_labels)
testData <- xgb.DMatrix(data = testData, label = testData_labels)

xgb2 <- xgb.train (params = tune_params, data = trainData, nrounds = 150, watchlist = list(val=testData,train=trainData), print_every_n = 50, early_stopping_rounds = 20, maximize = F , eval_metric = "merror")

xgbpred_train <- predict (xgb2, trainData)
confusionMatrix(factor(xgbpred_train), factor(trainData_labels)) # train accuracy

xgbpred_test <- predict (xgb2, testData)
confusionMatrix(factor(xgbpred_test), factor(testData_labels)) # test accuracy

## best model
train_xgb <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                          data = train[, c("funder_20","gps_height","install_3",
                                           "longitude","latitude","population","public_meeting",
                                           "scheme_management","permit","construction_year","extraction_type_class",
                                           "payment","quality_group",
                                           "quantity","source_type","waterpoint_type_group")])

test_xgb <- model.matrix(gps_height + longitude + latitude + population + construction_year ~ .-1, 
                         data = test[, c("funder_20","gps_height","install_3",
                                         "longitude","latitude","population","public_meeting",
                                         "scheme_management","permit","construction_year","extraction_type_class",
                                         "payment","quality_group",
                                         "quantity","source_type","waterpoint_type_group")])

train_labels_xgb <- as.numeric(train$status_group)-1

dtrain <- xgb.DMatrix(data = train_xgb,label = train_labels_xgb)
dtest <- xgb.DMatrix(data = test_xgb)
xgb3 <- xgb.train (params = tune_params, data = dtrain, nrounds = 460, watchlist = list(train=dtrain), print_every_n = 50, early_stopping_rounds = 20, maximize = F , eval_metric = "merror")
xgbpred_test <- predict (xgb2, dtest)

submission2 <- data.frame(test$id)
submission2$status_group <- ifelse(xgbpred_test == 0, "functional", (ifelse(xgbpred_test == 1, "functional needs repair", "non functional")))
names(submission2)[1] <- "id"
write.csv(submission2, "submission2.csv", row.names = FALSE, quote = FALSE)