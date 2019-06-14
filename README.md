# Instruction

## Background 
This project is developed based on a [bird sound detetection challenge](http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/). An algorithm based on logistic models was developed to determine where there are birds calling in a specific sound recording. The performance of this algorithms was further compared to a CNNs model. Detaild report can be found here. Due to the space limitation, only part of the data were used here to demonstrate two algorithms. Full dataset can be found in the link provided above. 

If applying to the new dataset is of interest, simply replace the data in the `soundData` folder! Have fun!

## Preparation
- Clone or download the [Bird-Sound-Detection-2019](https://github.com/SunnyTseng/Bird-Sound-Detection-2019) repo to the local computer
- Install R version 3.6.0
- Install required packages. 
  - For packages from CRAN.. 
  ```R
  install.packages("Name_of_Package")
  ```
  - For packages from Bioconductor, such as `EBImage`, please follow the [instruction](https://www.bioconductor.org/packages/release/bioc/html/EBImage.html). 
  ```R
  install.packages("BiocManager")
  BiocManager::install("Name_of_Package"), 
  ```

## Logistic Regression



## CNNs
