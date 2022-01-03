# Library
library(data.table)
library(caret)
library(xgboost)

# Load train and test
train <- fread('./project/volume/data/interim/train.csv')
test <- fread('./project/volume/data/interim/test.csv')

# Split id from train and test
id.train <- train$id
id.test <- test$id
train$id <- NULL
test$id <- NULL

# Split into y and x
y.train <- as.integer(as.factor(train$topic))-1
topics <- sort(unique(train$topic))
num.class <- length(topics)

dummies <- dummyVars(topic~., data=train)
x.train <- predict(dummies, newdata=train)
x.test <- predict(dummies, newdata=test)

# Make XGBosst objects
dtrain <- xgb.DMatrix(x.train, label=y.train, missing=NA)
dtest <- xgb.DMatrix(x.test, missing=NA)

# Initialize tuning log
tuning_log <- NULL

# Use for loop for parameter tuning
for (depth in 1:8)
  {
    # Set tuning parameters
    params <- list(booster          = 'gbtree',
                   tree_method      = 'hist',
                   objective        = 'multi:softprob',
                   num_class        = num.class,
                   # complexity
                   max_depth        = depth,
                   min_child_weight = 3,
                   gamma            = 0.2,
                   # diversity
                   eta              = 0.12,
                   subsample        = 0.73,
                   colsample_bytree = 1
    )

    # Preform cross-validaton for parameter tuning
    XGBm <- xgb.cv(params                = params,
                   data                  = dtrain,
                   missing               = NA,
                   nfold                 = 5,
                   # diversity
                   nrounds               = 10000,
                   early_stopping_rounds = 25,
                   # whether it shows error at each round
                   verbose               = 1
    )
    
    # Save parameters
    tuning_new <- data.table(t(params))
    
    # Save the best number of rounds
    best_nrounds <- unclass(XGBm)$best_iteration
    tuning_new$best_nrounds <- best_nrounds
    
    # Save the test set error
    error_cv <- unclass(XGBm)$evaluation_log[best_nrounds,]$test_mlogloss_mean
    tuning_new$error_cv <- error_cv
    
    # Add to the tuning log
    tuning_log <- rbind(tuning_log,tuning_new)
  }

# Use parameters with the smallest cross-validation error and best number of rounds
tuning_best <- tuning_log[which.min(tuning_log$error_cv),]
params <- list(booster          = 'gbtree',
               tree_method      = 'hist',
               objective        = 'multi:softprob',
               num_class        = num.class,
               max_depth        = tuning_best$max_depth,
               min_child_weight = tuning_best$min_child_weight,
               gamma            = tuning_best$gamma,
               eta              = tuning_best$eta,
               subsample        = tuning_best$subsample,
               colsample_bytree = tuning_best$colsample_bytree
)
nrounds <- tuning_best$best_nrounds

# Define watchlist and fit model
watchlist <- list(train=dtrain)
XGBm <- xgb.train(params        = params,
                  data          = dtrain,
                  missing       = NA,
                  nrounds       = nrounds,
                  print_every_n = TRUE,
                  watchlist     = watchlist
)

# Predict test with model
pred <- predict(XGBm, newdata=dtest, reshape=T)

# Save model
xgb.save(XGBm,"./project/volume/models/model.model")

# Make submission file and save
submit <- data.table(id=id.test)
submit <- cbind(submit, pred)
setnames(submit, paste0('V',1:num.class), topics)
fwrite(submit, "./project/volume/data/processed/submit.csv")
