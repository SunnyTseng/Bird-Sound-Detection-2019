rm(list=ls())


#####
# Library
#####
library(here)
library(tidyverse)
library(tuneR)
library(seewave)
library(filesstrings)

#####
# Percentiles + logistic regression
#####
getPercentiles <- function(dirPath, labelsExist=T){
  percentile=c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99)
  files <- list.files(dirPath)
  itemid <- c()
  features <- data_frame()
  for(i in files){
    sound <- readWave(paste0(dirPath, "/", i))
    freq <- 1000* dfreq(sound, wl=512, ovlp=50, bandpass=c(400, 10000))

    temp <- c()
    for(j in percentile){
      temp <- temp %>%
        c(., quantile(freq[,2], j))
    }
    itemid <- c(itemid, before_last_dot(i))
    features <- rbind(features, temp)
  }
  names(features) <- paste0("P", c(percentile))
  final <- data_frame(itemid=itemid) %>%
    cbind(., features) %>%
    mutate(database=ifelse(nchar(itemid)> 10, "warblrb10k", "ff1010bird"))
  
  
  if(labelsExist){
    dirPath_csv <- paste0(here("soundData", paste0(strsplit(dirPath, split="soundData/")[[1]][2], ".csv")))
    answerData <- read.csv(dirPath_csv, header=T) ## read in the answer
    final_answer <- final %>%                            ## append the answer based on the 
      left_join(answerData, by="itemid")  ## sequence of the specList extract hasbird yes=1, no=0
    final_answer$hasbird <- as.factor(final_answer$hasbird)
    final_answer$database <- as.factor(final_answer$database)
    
    return(final_answer)
  }else{
    return(final)
  }
}

# Takes about 1 hour
print(paste("Start extracting train features", Sys.time()))
trainData <- getPercentiles(dirPath=here("soundData/train/"), labelsExist=T)
print(paste("End extracting train features", Sys.time()))

# Takes slightly less
print(paste("Start extracting test features", Sys.time()))
testData <- getPercentiles(dirPath=here("soundData/test/"), labelsExist=T)
print(paste("End extracting test features", Sys.time()))


# Save / load
save(trainData, testData, file = "birdSoundData.RData")

##### Fit Logistic regression (bird.GLM.R5)#####
model <- glm(formula= hasbird~ (P0.3+ P0.5+ P0.7+ P0.9+ P0.95+ P0.975+ P0.99)*database, 
                   family=binomial(link="logit"), data=trainData)

# Compute probabilities and predictions on test set
probabilities <- predict(model, testData, type="response")

### Test accuracy
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
out <- discrimination(modelOut=probabilities, answer=testData$hasbird)
#auc <- MESS::auc(out %>% pull(sensitivity)/100, out %>% pull(FP)/100)


# Save model
save(model, file = "Logisticmodel.RData")

