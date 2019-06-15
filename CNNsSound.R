rm(list=ls())

### Reference: (1) CIFAR 10 example. (https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)
###            (2) Francisco Lima. CNN in R. (https://www.r-bloggers.com/convolutional-neural-networks-in-r/)


##### Libraries #####
library(keras)
library(EBImage)
library(stringr)
library(pbapply)

library(here)
library(tidyverse)
library(tuneR)
library(seewave)
library(phonTools)
library(BBmisc)

##### Process image #####
# Set image size
extract_feature <- function(dirPath, width, height, labelsExist=T) {
  
  ## get the featuress list (as vectors)
  getSpectrogram <- function(dirPath){
        #Input: dirPath string, takes path for train or test
        #Output: a list with spectrograms arrays accordingly to the input, list name is the file name
        #Reference: https://hansenjohnson.org/post/spectrograms-in-r/
        featureList <- list()
        i <- 1
        wl <- 6
        timestep <- -1000
        img_size <- width * height
        soundNames <- paste0(dirPath, "/", list.files(dirPath))
        for (soundName in soundNames){
          spec <- loadsound(soundName) %>%
            phonTools::spectrogram(., colors=F, show=F, windowlength=wl, timestep=timestep, window="hann") %>%
            .$spectrogram %>%
            BBmisc::normalize(method="range") %>%
            as.Image()
          
          img_vector <- spec %>%
            resize(., w=width, h=height) %>%  ## Resize image
            .@.Data %>%                       ## Get the image as a matrix
            t() %>%                           ## Coerce to a vector (row-wise)
            as.vector()
          
          featureList[[i]] <- img_vector
          names(featureList)[i] <- strsplit(tail(strsplit(soundName, split="/")[[1]], 1), split=".wav")[[1]]
          i = i+1
        }
      return(featureList)
  }
  
  img_size <- width * height
  ## bind the list of vector into matrix
  featureList <- getSpectrogram(dirPath=dirPath)
  feature_matrix <- do.call(rbind, featureList) %>%
    as.data.frame(.)
  names(feature_matrix) <- paste0("pixel", c(1:img_size))


  ## produce an answer list for accuracy calculation 
  if(labelsExist){
    dirPath_csv <- paste0(here("soundData",
                               paste0(strsplit(dirPath, split="soundData/")[[1]][2], ".csv")))
    answerData <- read.csv(dirPath_csv, header=T) ## read in the answer
    y <- data.frame(itemid=names(featureList)) %>%   ## append the answer based on the 
      left_join(answerData, by="itemid") %>%      ## sequence of the specList
      .$hasbird                                   ## extract hasbird yes=1, no=0
    return(list(X = feature_matrix, y = y))
  }else{
    return(feature_matrix)
    }
}


width <- 50
height <- 50
# Takes approx. 7 hours
print(paste("Start extracting train features", Sys.time()))
trainData <- extract_feature(here("soundData/train/"), width, height, labelsExist=T)
print(paste("End extracting train features", Sys.time()))

# Takes approx. 1.5 hours
print(paste("Start extracting test features", Sys.time()))
testData <- extract_feature(here("soundData/test/"), width, height, labelsExist=T)
print(paste("End extracting test features", Sys.time()))



# Save / load
save(trainData, testData, file = "birdSoundData.RData")
# load("birdSoundData.RData")


##### Fit NN #####
# Fix structure for 2d CNN
train_array <- t(trainData$X)
dim(train_array) <- c(50, 50, nrow(trainData$X), 1)
# Reorder dimensions
train_array <- aperm(train_array, c(3,1,2,4))

test_array <- t(testData$X)
dim(test_array) <- c(50, 50, nrow(testData$X), 1)
# Reorder dimensions
test_array <- aperm(test_array, c(3,1,2,4))


# Build CNN model
model <- keras_model_sequential() 
model %>% 
      layer_conv_2d(kernel_size = c(3, 3), filter = 32,
                    activation = "relu", padding = "same",
                    input_shape = c(50, 50, 1),
                    data_format = "channels_last") %>%
      layer_conv_2d(kernel_size = c(3, 3), filter = 32,
                    activation = "relu", padding = "valid") %>%
      layer_max_pooling_2d(pool_size = 2) %>%
      layer_dropout(rate = 0.25) %>%
      
      layer_conv_2d(kernel_size = c(3, 3), filter = 64, strides = 2,
                    activation = "relu", padding = "same") %>%
      layer_conv_2d(kernel_size = c(3, 3), filter = 64,
                    activation = "relu", padding = "valid") %>%
      layer_max_pooling_2d(pool_size = 2) %>%
      layer_dropout(rate = 0.25) %>%
      
      layer_flatten() %>%
      layer_dense(units = 50, activation = "relu") %>% 
      layer_dropout(rate = 0.25) %>%
      layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
      loss = 'binary_crossentropy',
      optimizer = "adam",
      metrics = c('accuracy')
)

history <- model %>% fit(
      x = train_array, y = as.numeric(trainData$y), 
      epochs = 30, batch_size = 100, 
      validation_split = 0.2
)

plot(history)

# Compute probabilities and predictions on test set
predictions <-  predict_classes(model, test_array)
probabilities <- predict_proba(model, test_array)



discrimination <- function(modelOut, answer){
  cutoffs <- seq(0.3, 0.9, 0.05)
  accuracy <- c()
  FP <- c()
  FN <- c()
  #Kappa <- c()
  for (i in seq(along=cutoffs)){
    Ctable <- table(answer, ifelse(modelOut >= cutoffs[i], 1, 0))
    accuracy <- c(accuracy, (100*(Ctable[1]+Ctable[4])/sum(Ctable)))
    FP <- c(FP, 100*Ctable[3]/(Ctable[1]+ Ctable[3]))
    FN <- c(FN, 100*Ctable[2]/(Ctable[2]+ Ctable[4]))
    #Kappa <- c(Kappa, Kappa.test(Ctable)$Result[2]$estimate*100)
  }
  
  discrime <- data.frame(cutoff=cutoffs, accuracy=accuracy, FP=FP, FN=FN)
  discrime$sensitivity <- 100-discrime$FP
  discrime$specificity <- 100-discrime$FN
  
  return(discrime)
}
### Test accuracy
out <- discrimination(modelOut=probabilities, answer=testData[[2]])


# Save model
save(model, file = "CNNmodel.RData")
