
# Set Working Directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Open codes
file.edit('./project/required/requirements.R')
file.edit('./project/src/features/build_features.R')
file.edit('./project/src/models/train_model.R')

# Run codes 
source('./project/required/requirements.R')
source('./project/src/features/build_features.R')
source('./project/src/models/train_model.R')
