# Library
library(data.table)
library(httr)
library(Rtsne)
library(ggplot2)
library(gridExtra)

# Load train and test
train <- fread('./project/volume/data/raw/train.csv')
test <- fread('./project/volume/data/raw/test.csv')

# Reformat subreddits into one response
topics <- names(train[,-c('id','text')])
train$topic <- 'NA'
for(i in 1:length(topics)) train[get(topics[i])==1]$topic <- topics[i]
train[,topics] <- NULL

# Combine train and test into master
train$train <- 1
test$train <- 0
master <- rbind(train, test, fill=T)

# Embedding
train_emb <- fread('./project/volume/data/raw/train_emb.csv')
test_emb <- fread('./project/volume/data/raw/test_emb.csv')
emb_dt <- rbind(train_emb, test_emb)

# Dimension reduction by t-SNE 
tsne <- Rtsne(emb_dt, dim=3, check_duplicates=F, perplexity=3000, theta=0.4)

# Create data table from t-SNE
tsne_dt <- data.table(tsne$Y)
tsne_dt$topic <- master$topic

# Visualization
grid.arrange(ggplot(tsne_dt[master$train==1],aes(x=V1,y=V2,col=topic))+geom_point(),
             ggplot(tsne_dt[master$train==1],aes(x=V1,y=V3,col=topic))+geom_point(),
             ggplot(tsne_dt[master$train==1],aes(x=V2,y=V3,col=topic))+geom_point(),
             nrow=1)

# Split master back into train and test
tsne_dt$id <- master$id
train <- tsne_dt[master$train==1, .(id,V1,V2,V3,topic)]
test <- tsne_dt[master$train==0, .(id,V1,V2,V3,topic)]

# Save train and test
fwrite(train, './project/volume/data/interim/train.csv')
fwrite(test, './project/volume/data/interim/test.csv')
