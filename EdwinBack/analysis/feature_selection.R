library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)

# tips
# xgboost, make sure you tune the learning rate and pick the right one
setwd("~/Desktop/NYCDSA/Projects/proj3_ml/house-prices-advanced-regression-techniques")

house_train = read.csv('train.csv', stringsAsFactors = F)
# house_train = house_train %>% select(., -X)
house_test = read.csv('test.csv', stringsAsFactors = F)
# house_test = house_test %>% select(., -X)

#Getting rid of the IDs but keeping the test IDs in a vector. These are needed to compose the submission file
test_labels <- house_test$Id
house_test$Id <- NULL
house_train$Id <- NULL

house_test$SalePrice <- NA
all = rbind(house_train, house_test)
dim(all) # 79 predictors and 1 response variable SalePrice

ggplot(data=all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(fill="blue", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

summary(all$SalePrice)

##############################################################################################
############################### Most Important Numeric Variables #############################
##############################################################################################

# Correlations with SalePrice
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
cat('There are', length(numericVars), 'numeric variables')

factorVarNames = names(factorVars)

all_numVar = all[ , numericVars] # only numeric varaibles
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") # correlations of all numeric variables

# CORRELATION MATRIX PLOT
# sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
# select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
# plot
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")

# Top Correlated features
# OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), GarageArea (0.62),
# TotalBsmtSF (0.61), X1stFlrSF (0.61), FullBath (0.56), TotRmsAbvGrd (0.53), YearBuilt (0.52), YearRemodAdd (0.51) 

### NUMERIC FEATURES ###

# OverallQual (Top Correlated Feature)
# Boxplot Distribution
ggplot(data=all[!is.na(all$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
  geom_boxplot(col='blue') + labs(x='Overall Quality') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

# GrLivArea (Second Top Correlated Feature)
# Scatterplot with regression line (Notice 2 outliers in 524 & 1299)
ggplot(data=all[!is.na(all$SalePrice),], aes(x=GrLivArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)]>4500, rownames(all), '')))

all[c(524, 1299), c('SalePrice', 'GrLivArea', 'OverallQual')]

# GarageCars (Third top correlated feature)
ggplot(data=all[!is.na(all$SalePrice),], aes(x=factor(GarageCars), y=SalePrice))+
  geom_boxplot(col='red') + labs(x='GarageCars') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

####################################################
#################### FIND MISSING DATA ####################
####################################################

NAcol <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)

cat('There are', length(NAcol), 'columns with missing values')

####################################################
#################### IMPUTE ####################
####################################################

#### PoolQC, Values are ordinal so we convert to integers
all$PoolQC[is.na(all$PoolQC)] = 'None' # Change NA to None
Qualities = c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5) # character vector
all$PoolQC = as.integer(revalue(all$PoolQC, Qualities))
table(all$PoolQC)

#### PoolArea for PoolQC == 0, impute based on the overall quality of each of the three houses since there is no obvious
# correlation between the PoolQC and PoolArea i.e. higher PoolQC = larger PoolArea
all[all$PoolArea > 0 & all$PoolQC == 0, c('PoolArea', 'PoolQC', 'OverallQual')]
# OverallQual is out of 10, PoolQC out of 5 so the score is halved
all$PoolQC[2421] = 2
all$PoolQC[2504] = 3
all$PoolQC[2600] = 2

#### MiscFeature, Values are NOT ordinal -> convert from char to factor
all$MiscFeature[is.na(all$MiscFeature)] = 'None'
all$MiscFeature = as.factor(all$MiscFeature)

# Plot of results
ggplot(all[!is.na(all$SalePrice),], aes(x=MiscFeature, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))
# Table of results
table(all$MiscFeature)

# Having a shed probably means ‘no Garage’, which would explain the lower sales price for Shed. 
# Also, while it makes a lot of sense that a house with a Tennis court is expensive, there is 
# only one house with a tennis court in the training set.

#### Alley, Values are NOT ordinal -> convert to factors
all$Alley[is.na(all$Alley)] = 'None'
all$Alley = as.factor(all$Alley)

ggplot(all[!is.na(all$SalePrice),], aes(x=Alley, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 200000, by=50000), labels = comma)

table(all$Alley)

#### Fence, Values seem ordinal at first but a deeper look shows that they are not -> convert to factors
# Can't really compare privacy and wood quality
all$Fence[is.na(all$Fence)] = 'None'
# Higher privacy/wood != higher price
all[!is.na(all$SalePrice), ] %>% 
  group_by(Fence) %>% 
  summarise(median = median(SalePrice), counts = n())
# Values do not seem ordinal (no fence is best). Therefore, convert Fence into a factor.
all$Fence = as.factor(all$Fence)

#### FireplaceQu, Values are ordinal using same chars as PoolQC
all$FireplaceQu[is.na(all$FireplaceQu)] = 'None'
all$FireplaceQu = as.integer(revalue(all$FireplaceQu, Qualities))

#### Fireplaces values are int and have NO missing values, leave it alone

#### LotFrontage. Values are numeric floats
# A reasonable imputation seems to be to take the median per neigborhood.
ggplot(all[!is.na(all$LotFrontage),], aes(x=as.factor(Neighborhood), y=LotFrontage)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

for (i in 1:nrow(all)){
  if(is.na(all$LotFrontage[i])){
    all$LotFrontage[i] <- as.double(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]], na.rm=TRUE)) 
  }
}

class(as.double(all$LotFrontage))
sum(is.na(all$LotFrontage))

#### LotShape, no missing values but values seem to be ordinal enough with regular = best, irregular = worst
sum(is.na(all$LotShape)) # 0
# 1 = irregular, 4 = regular
all$LotShape = as.integer(revalue(all$LotShape, c('IR3' = 1, 'IR2' = 2, 'IR1' = 3, 'Reg' = 4)))
table(all$LotShape)

# LotConfig, no missings, not ordinal
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(LotConfig), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

all$LotConfig = as.factor(all$LotConfig)
table(all$LotConfig)

# Three Garage Variables with NA's = GarageCars, GarageArea, GarageType
# Replace all 159 missing GarageYrBlt: Year garage was built values with the values in YearBuilt
all$GarageYrBlt[is.na(all$GarageYrBlt)] <- all$YearBuilt[is.na(all$GarageYrBlt)]

# find out where the differences between the 157 NA GarageType and the other 
# 3 character variables with 159 NAs come from
# check if all 157 NAs are the same observations among the variables with 157/159 NAs
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))

# Find the 2 additional NAs
kable(all[!is.na(all$GarageType) & is.na(all$GarageFinish), c('GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])

# house 2127 actually does seem to have a Garage and house 2577 does not.
# Imputing modes.
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1]
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]

#display "fixed" house
kable(all[2127, c('GarageYrBlt', 'GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])

#fixing 3 values for house 2577
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- NA

#check if NAs of the character variables are now all 158
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))

### GarageType
all$GarageType[is.na(all$GarageType)] = 'No Garage'
all$GarageType = as.factor(all$GarageType)
table(all$GarageType)

### GarageFinish
all$GarageFinish[is.na(all$GarageFinish)] = 'None'
Finish = c('None' = 0, 'Unf' = 1, 'RFn' = 2, 'Fin' = 3)

all$GarageFinish = as.integer(revalue(all$GarageFinish, Finish))
table(all$GarageFinish)

### GarageQual, GarageCond
all$GarageQual[is.na(all$GarageQual)] = 'None'
all$GarageQual = as.integer(revalue(all$GarageQual, Qualities))
table(all$GarageQual)

all$GarageCond[is.na(all$GarageCond)] = 'None'
all$GarageCond = as.integer(revalue(all$GarageCond, Qualities))
table(all$GarageCond)

### Basement Variables ####
#check if all 79 NAs are the same observations among the variables with 80+ NAs
length(which(is.na(all$BsmtQual) & is.na(all$BsmtCond) & is.na(all$BsmtExposure) & is.na(all$BsmtFinType1) & is.na(all$BsmtFinType2)))

#Find the additional NAs; BsmtFinType1 is the one with 79 NAs
all[!is.na(all$BsmtFinType1) & (is.na(all$BsmtCond)|is.na(all$BsmtQual)|is.na(all$BsmtExposure)|is.na(all$BsmtFinType2)), c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]

#Imputing modes to fix the 9 houses found above that have missings when BsmtFinType1 not NA
all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]
all$BsmtQual[c(2218, 2219)] <- names(sort(-table(all$BsmtQual)))[1]

# Now that the 5 variables considered agree upon 79 houses with ‘no basement’, 
# I am going to factorize/hot encode them below.

#### BsmtQual, ordinal convert to integer
all$BsmtQual[is.na(all$BsmtQual)] = 'None'
all$BsmtQual = as.integer(revalue(all$BsmtQual, Qualities))
table(all$BsmtQual)

#### BsmtCond, ordinal convert to integer
all$BsmtCond[is.na(all$BsmtCond)] = 'None'
all$BsmtCond = as.integer(revalue(all$BsmtCond, Qualities))
table(all$BsmtCond)

#### BsmtExposure
all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)

all$BsmtExposure<-as.integer(revalue(all$BsmtExposure, Exposure))
table(all$BsmtExposure)

#### BsmtFinType1
all$BsmtFinType1[is.na(all$BsmtFinType1)] = 'None'
FinType = c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)

all$BsmtFinType1<-as.integer(revalue(all$BsmtFinType1, FinType))
table(all$BsmtFinType1)

#### BsmtFinType2
all$BsmtFinType2[is.na(all$BsmtFinType2)] = 'None'

all$BsmtFinType2 = as.integer(revalue(all$BsmtFinType2, FinType))
table(all$BsmtFinType2)

#display remaining NAs. Using BsmtQual as a reference for the 79 houses without basement agreed upon earlier
all[(is.na(all$BsmtFullBath)|is.na(all$BsmtHalfBath)|is.na(all$BsmtFinSF1)|is.na(all$BsmtFinSF2)|is.na(all$BsmtUnfSF)|is.na(all$TotalBsmtSF)), c('BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')]

#### BsmtFullBath
all$BsmtFullBath[is.na(all$BsmtFullBath)] = 0
table(all$BsmtFullBath)

#### BsmtHalfBath
all$BsmtHalfBath[is.na(all$BsmtHalfBath)] = 0
table(all$BsmtHalfBath)

#### BsmtFinType1
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] = 0

#### BsmtFinType2
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] = 0

#### BsmtUnfSF
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] = 0

#### TotalBsmtSF
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] = 0

# Masonry Variables
length(which(is.na(all$MasVnrType) & is.na(all$MasVnrArea))) #23 missings

#find the house that should have a MasVnrType
all[(all$MasVnrType == 'None') & (all$MasVnrArea > 0), c('MasVnrType', 'MasVnrArea')]

#fix this veneer type by imputing the mode
all$MasVnrType[2611] = names(sort(-table(all$MasVnrType)))[2] #taking the 2nd value as the 1st is 'none'
all[2611, c('MasVnrType', 'MasVnrArea')] #BrkFace

#### MasVnrType, not ordinal, convert to factors
all$MasVnrType[is.na(all$MasVnrType)] = 'None'
all[!is.na(all$SalePrice),] %>% group_by(MasVnrType) %>% summarise(median = median(SalePrice), counts=n()) %>% arrange(median)

# does not seem ordinal enough, just convert to factor
all$MasVnrType = as.factor(all$MasVnrType)
table(all$MasVnrType)

#### MasVnrArea, fill missings with zeros
all$MasVnrArea[is.na(all$MasVnrArea)] = 0

### MSZoning, not ordinal, categorical, 4 NA's
#imputing the mode
all$MSZoning[is.na(all$MSZoning)] = names(sort(-table(all$MSZoning)))[1] # no nones, so use top sorted value
all$MSZoning = as.factor(all$MSZoning)
table(all$MSZoning)

### KitchenQual
all$KitchenQual[is.na(all$KitchenQual)] <- 'TA' #replace with most common value
all$KitchenQual<-as.integer(revalue(all$KitchenQual, Qualities))
table(all$KitchenQual)

### KitchenAbvGr has no NA's
table(all$KitchenAbvGr)

### Utilities, 2 NA's
table(all$Utilities) # basically all of the values are that houses have all public utilities
# essentially, this column isn't telling us much so we can get rid of it
kable(all[is.na(all$Utilities) | all$Utilities=='NoSeWa', 1:9])
all$Utilities = NULL

#### Functionality, 2 NA, seems to be ordinal with Typ = best, Sal = worst
# Impute the mode (Typ) for the 2 NA's
all$Functional[is.na(all$Functional)] = names(sort(-table(all$Functional)))[1]

# Convert categorical to ordinal integers
all$Functional = as.integer(revalue(all$Functional, c('Sal' = 0, 'Sev' = 1, 'Maj2' = 2, 'Maj1' = 3, 'Mod' = 4, 'Min1' = 5, 'Min2' = 6, 'Typ' = 7)))
table(all$Functional)

#### Exterior Variables (Exterior1st, Exterior2nd, ExterQual, ExterCond)

#### Exterior1st and Exterior2nd have same missing value row, no ordinality, convert to factors
# Exterior1st, Impute by mode which is VinylSd
all$Exterior1st[is.na(all$Exterior1st)] = names(sort(-table(all$Exterior1st)))[1]
all$Exterior1st = as.factor(all$Exterior1st)
table(all$Exterior1st)

#### Exterior2nd, Impute by mode again
all$Exterior2nd[is.na(all$Exterior2nd)] = names(sort(-table(all$Exterior2nd)))[1]
all$Exterior2nd = as.factor(all$Exterior2nd)
table(all$Exterior2nd)

#### ExterQual, no missings, convert directly to integer using Qualities
all$ExterQual = as.integer(revalue(all$ExterQual, Qualities))
table(all$ExterQual)

#### ExterCond, no missings, convert directly to integer using Qualities
all$ExterCond = as.integer(revalue(all$ExterCond, Qualities))
table(all$ExterCond)

#### Electrical, 1 missing, can't have no electrical so we impute NA with mode
all$Electrical[is.na(all$Electrical)] = names(sort(-table(all$Electrical)))
# convert to factor
all$Electrical = as.factor(all$Electrical)
table(all$Electrical)

#### SaleType (1 missing) and SaleCondition (no missing)
all$SaleType[is.na(all$SaleType)] = names(sort(-table(all$SaleType)))
all$SaleType = as.factor(all$SaleType)
table(all$SaleType)

#### SaleCondition not ordinal, convert to factor
all$SaleCondition = as.factor(all$SaleCondition)
table(all$SaleCondition)


#############################################################################################
#################### LABEL ENCODING/FACTORIZING 15 REMAINING CHAR VARIABLES ####################
#############################################################################################

# No missing values in these features
Charcols = names(all[,sapply(all, is.character)])
Charcols
cat('There are', length(Charcols), 'remaining columns with character values')

#### Foundation, no ordinality so convert to factor
all$Foundation = as.factor(all$Foundation)
table(all$Foundation)
  
#### Heating, HeatingQC, can't have no heating; HeatingQC is ordinal, Heating is not
all$Heating = as.factor(all$Heating)
table(all$Heating)

all$HeatingQC = as.integer(revalue(all$HeatingQC, Qualities))
table(all$HeatingQC)

#### CentralAir, convert yes/no to 1/0
all$CentralAir = as.integer(revalue(all$CentralAir, c('N' = 0, 'Y' = 1)))
table(all$CentralAir)

#### RoofStyle, RoofMatl, both not ordinal, convert to factors
all$RoofStyle = as.factor(all$RoofStyle)
table(all$RoofStyle)

all$RoofMatl = as.factor(all$RoofMatl)
table(all$RoofMatl)

#### LandContour (not ordinal), LandSlope (ordinal)
all$LandContour = as.factor(all$LandContour)
table(all$LandContour)

all$LandSlope = as.integer(revalue(all$LandSlope, c('Sev' = 0, 'Mod' = 1, 'Gtl' = 2)))
table(all$LandSlope)

#### BldgType, HouseStyle
# check to see if BldgType is ordinal; if 1-Fam detached has highest price and TwnhsI is lowest -> NOT TRUE
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(BldgType), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..))

all$BldgType = as.factor(all$BldgType)
table(all$BldgType)

all$HouseStyle = as.factor(all$HouseStyle)
table(all$HouseStyle)

#### Neighborhood , Condition1, Condition2 (no ordinality for all three features)
all$Neighborhood = as.factor(all$Neighborhood)
table(all$Neighborhood)

all$Condition1 = as.factor(all$Condition1)
table(all$Condition1)

all$Condition2 = as.factor(all$Condition2)
table(all$Condition2)

#### Pavement of street and driveway, majority of street access and driveways are paved
all$Street = as.integer(revalue(all$Street, c('Grvl' = 0, 'Pave' = 1)))
table(all$Street)

all$PavedDrive = as.integer(revalue(all$PavedDrive, c('N' = 0, 'P' = 1, 'Y' = 2)))
table(all$PavedDrive)

#############################################################################################
#################### CHANGING SOME NUMERICS INTO FACTORS ####################
#############################################################################################
str(all$YrSold)
str(all$MoSold)
all$MoSold = as.factor(all$MoSold)

# SalePrice by YrSold
ys <- ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(YrSold), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice

# SalePrice by MoSold
ms <- ggplot(all[!is.na(all$SalePrice),], aes(x=MoSold, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue')+
  scale_y_continuous(breaks= seq(0, 800000, by=25000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..)) +
  coord_cartesian(ylim = c(0, 200000)) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice

grid.arrange(ys, ms, widths=c(1,2))

#### MSSubClass
all$MSSubClass = as.factor(all$MSSubClass)

#revalue for better readability
all$MSSubClass = revalue(all$MSSubClass, c('20'='1 story 1946+', '30'='1 story 1945-', '40'='1 story unf attic', '45'='1,5 story unf', '50'='1,5 story fin', '60'='2 story 1946+', '70'='2 story 1945-', '75'='2,5 story all ages', '80'='split/multi level', '85'='split foyer', '90'='duplex all style/age', '120'='1 story PUD 1946+', '150'='1,5 story PUD all', '160'='2 story PUD 1946+', '180'='PUD multilevel', '190'='2 family conversion'))

str(all$MSSubClass)

#### SaleCondition
all$SaleCondition = as.character(all$SaleCondition)
all = all[(all$SaleCondition != 'Abnorml') & (all$SaleCondition != 'Family'), ]
all$SaleCondition = as.factor(all$SaleCondition)

#############################################################################################
#################### CHANGING SOME NUMERICS INTO FACTORS ####################
#############################################################################################
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
factorVars <- which(sapply(all, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')
# made MasVnrType == factor instead of ordinal integer, so 55 numerics and 24 factors

# CORRELATION MATRIX PLOT
all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

# Variable importance using a quick simple RandomForest
set.seed(2018)
quick_RF <- randomForest(x = all[1:1460, -79], y = all$SalePrice[1:1460], ntree = 100, importance = TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) +
  geom_bar(stat = 'identity') +
  labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') +
  coord_flip() + theme(legend.position="none")

# 
s1 <- ggplot(data = all, aes(x=GrLivArea)) +
  geom_density() + labs(x='Square feet living area')
s2 <- ggplot(data =all, aes(x=as.factor(TotRmsAbvGrd))) +
  geom_histogram(stat='count') + labs(x='Rooms above Ground')
s3 <- ggplot(data = all, aes(x=X1stFlrSF)) +
  geom_density() + labs(x='Square feet first floor')
s4 <- ggplot(data = all, aes(x=X2ndFlrSF)) +
  geom_density() + labs(x='Square feet second floor')
s5 <- ggplot(data = all, aes(x=TotalBsmtSF)) +
  geom_density() + labs(x='Square feet basement')
s6 <- ggplot(data = all[all$LotArea<100000,], aes(x=LotArea)) +
  geom_density() + labs(x='Square feet lot')
s7 <- ggplot(data = all, aes(x=LotFrontage)) +
  geom_density() + labs(x='Linear feet lot frontage')
s8 <- ggplot(data = all, aes(x=LowQualFinSF)) +
  geom_histogram() + labs(x='Low quality square feet 1st & 2nd')

layout <- matrix(c(1,2,5,3,4,8,6,7),4,2,byrow=TRUE)
multiplot(s1, s2, s3, s4, s5, s6, s7, s8, layout=layout)

cor(all$GrLivArea, (all$X1stFlrSF + all$X2ndFlrSF + all$LowQualFinSF))
head(all[all$LowQualFinSF>0, c('GrLivArea', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF')])

# Neighborhood Sale Price and Count
n1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=Neighborhood, y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
n2 <- ggplot(data=all, aes(x=Neighborhood)) +
  geom_histogram(stat='count')+
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(n1, n2)

# Overall Quantity
q1 <- ggplot(data=all, aes(x=as.factor(OverallQual))) +
  geom_histogram(stat='count')
q2 <- ggplot(data=all, aes(x=as.factor(ExterQual))) +
  geom_histogram(stat='count')
q3 <- ggplot(data=all, aes(x=as.factor(BsmtQual))) +
  geom_histogram(stat='count')
q4 <- ggplot(data=all, aes(x=as.factor(KitchenQual))) +
  geom_histogram(stat='count')
q5 <- ggplot(data=all, aes(x=as.factor(GarageQual))) +
  geom_histogram(stat='count')
q6 <- ggplot(data=all, aes(x=as.factor(FireplaceQu))) +
  geom_histogram(stat='count')
q7 <- ggplot(data=all, aes(x=as.factor(PoolQC))) +
  geom_histogram(stat='count')

layout <- matrix(c(1,2,8,3,4,8,5,6,7),3,3,byrow=TRUE)
multiplot(q1, q2, q3, q4, q5, q6, q7, layout=layout)

#########################################################################################
################################## FEATURE ENGINEERING ##################################
#########################################################################################

# Total Number of Bathrooms
all$TotBathrooms <- all$FullBath + (all$HalfBath*0.5) + all$BsmtFullBath + (all$BsmtHalfBath*0.5)

tb1 <- ggplot(data=all[!is.na(all$SalePrice),], aes(x=as.factor(TotBathrooms), y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)
tb2 <- ggplot(data=all, aes(x=as.factor(TotBathrooms))) +
  geom_histogram(stat='count')
grid.arrange(tb1, tb2)

### House Age, Remodeled (Yes or No)
all$Remod <- ifelse(all$YearBuilt==all$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
all$Age <- as.numeric(all$YrSold)-all$YearRemodAdd

ggplot(data=all[!is.na(all$SalePrice),], aes(x=Age, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

cor(all$SalePrice[!is.na(all$SalePrice)], all$Age[!is.na(all$SalePrice)])

# houses that are remodeled are valued less
ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(Remod), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed") #dashed line is median SalePrice

### 116 new houses
all$IsNew <- ifelse(all$YrSold==all$YearBuilt, 1, 0)
table(all$IsNew)

ggplot(all[!is.na(all$SalePrice),], aes(x=as.factor(IsNew), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed") #dashed line is median SalePrice

all$YrSold <- as.factor(all$YrSold) #the numeric version is now not needed anymore

### Location from OverallQual, OverallCond, ExterQual, ExterCond, Functional
all$HouseQC = ((all$OverallQual) + (all$OverallCond) + (all$ExterCond*2) + (all$ExterQual*2) + (all$Functional*(10/7)))/5
temp = all %>% group_by(Neighborhood) %>% summarise(n=n(), location = mean(HouseQC)) %>% arrange(desc(location))

### Railroad - Proximity to railroad (none, adjacent, or within 200'), from Condition1 and Condition2
all$Condition1 = as.character(all$Condition1)
all$Condition2 = as.character(all$Condition2)

all$Railroad1 = as.integer(revalue(all$Condition1, c('Norm' = 0, 'Feedr' = 0, 'PosN' = 0, 'PosA' = 0, 'Artery' = 1, 'RRAe' = 1, 'RRAn' = 1, 'RRNn' = 2, 'RRNe' = 2)))
all$Railroad2 = as.integer(revalue(all$Condition2, c('Norm' = 0, 'Feedr' = 0, 'PosN' = 0, 'PosA' = 0, 'Artery' = 1, 'RRAe' = 1, 'RRAn' = 1, 'RRNn' = 2, 'RRNe' = 2)))
all$Amenities1 = as.integer(revalue(all$Condition1, c('Norm' = 0, 'Feedr' = 0, 'PosN' = 1, 'PosA' = 1, 'Artery' = 0, 'RRAe' = 0, 'RRAn' = 0, 'RRNn' = 0, 'RRNe' = 0)))
all$Amenities2 = as.integer(revalue(all$Condition2, c('Norm' = 0, 'Feedr' = 0, 'PosN' = 1, 'PosA' = 1, 'Artery' = 0, 'RRAe' = 0, 'RRAn' = 0, 'RRNn' = 0, 'RRNe' = 0)))

# Railroad, Amenities
all$Railroad = ifelse((all$Railroad1 == 1) | (all$Railroad2 == 1), 1, ifelse((all$Railroad1 == 2) | (all$Railroad2 == 2), 2, 0))
all$Amenities = ifelse((all$Amenities1 == 1) | (all$Amenities2 == 1), 1, 0)

all$Condition1 = as.factor(all$Condition1)
all$Condition2 = as.factor(all$Condition2)

### Binning Neighborhood
nb1 <- ggplot(all[!is.na(all$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=median), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "median", fill='blue') + labs(x='Neighborhood', y='Median SalePrice') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
nb2 <- ggplot(all[!is.na(all$SalePrice),], aes(x=reorder(Neighborhood, SalePrice, FUN=mean), y=SalePrice)) +
  geom_bar(stat='summary', fun.y = "mean", fill='blue') + labs(x='Neighborhood', y="Mean SalePrice") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice
grid.arrange(nb1, nb2)

# based on mean saleprice for each neighborhood
all$NeighRich[all$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
all$NeighRich[!all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
all$NeighRich[all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0
table(all$NeighRich)

### Total Sqaure Feet
all$TotalSqFeet = all$GrLivArea + all$LowQualFinSF + all$TotalBsmtSF - all$BsmtUnfSF 

ggplot(data=all[!is.na(all$SalePrice),], aes(x=TotalSqFeet, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_text_repel(aes(label = ifelse(all$GrLivArea[!is.na(all$SalePrice)]>4500, rownames(all), '')))

# correlation with outliers
cor(all$SalePrice, all$TotalSqFeet, use= "pairwise.complete.obs")

# correlation much higher without these two outliers
cor(all$SalePrice[-c(524, 1299)], all$TotalSqFeet[-c(524, 1299)], use= "pairwise.complete.obs")

### Consolidating porch variables, leave woodDeck alone
all$TotalPorchSF = all$OpenPorchSF + all$EnclosedPorch + all$X3SsnPorch + all$ScreenPorch
cor(all$SalePrice, all$TotalPorchSF, use= "pairwise.complete.obs")

### Price per SF
all$PricePerSF = all$SalePrice / all$TotalSqFeet

ggplot(data=all[!is.na(all$SalePrice),], aes(x=TotalPorchSF, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=100000), labels = comma)

ggplot(data=all[!is.na(all$SalePrice),], aes(x=HouseQC, y=PricePerSF))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black", aes(group=1)) +
  scale_y_continuous(breaks= seq(0, 500, by=50), labels = comma)

# better neighborhood location might indicate higher prices
ggplot(all[!is.na(all$SalePrice),], aes(x=reorder(as.factor(Location), PricePerSF, FUN=mean), y=PricePerSF)) +
  geom_bar(stat='summary', fun.y = "mean", fill='blue') + labs(x='Location', y="Mean Price Per SF") +
  scale_y_continuous(breaks= seq(0, 200, by=25), labels = comma)


#########################################################################################
################################## PREPARING DATA FOR MODELING ##########################
#########################################################################################
# Drop variables that are:
# highly correlated (multicollinear), lots of missings (over 1000), associated features
all = all[!c(524, 1299),] # remove outliers for high GrLivArea and low SalePrice
all = all[!(all$MSZoning == 'C (all)'), ] # remove 25 observations with zoning type C (all) 

numericVars = which(sapply(all, is.numeric))
numericVarNames = names(numericVars)
numericVarNames <-
  numericVarNames[(
    numericVarNames %in% c(
      'LotFrontage',
      'LotArea',
      'BedroomAbvGr',
      'Fireplaces',
      'GarageArea',
      'WoodDeckSF',
      'Age',
      'HouseQC',
      'TotBathrooms',
      'TotalSqFeet',
      'TotalPorchSF',
      'PricePerSF',
      'SalePrice'
    )
  )] #numericVarNames was created before having done anything
DFnumeric <- all[, (names(all) %in% numericVarNames)]

DFfactors <- all[ , !(names(all) %in% numericVarNames)]
DFfactors <- DFfactors[ , (names(DFfactors) != 'SalePrice') & (names(DFfactors) != 'PricePerSF')]

dropFactVars = c(
  'Alley', 'Fence', 'MiscFeature', 'MiscVal', # Too many NA's
  'HouseStyle', # redundant with MSZoning or MSSubClass
  'Condition1', 'Condition2', 'Railroad1', 'Railroad2', # Used to define Railroad proximity
  'YearBuilt', # Used in Remod, IsNew
  'YrSold', 'MoSold', # Used in Age
  'YearRemodAdd', # Used in Age, Remod
  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', # Explains TotalBsmtSF
  'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', # Explains GrLivArea
  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', # Explains TotBathrooms
  'TotRmsAbvGrd', # Multicolinearity
  'GarageYrBlt', 'GarageCars', 'GarageCond', # multicolinear
  'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', # Explains TotalPorchSF
  'PoolArea', 'PoolQC', # PoolQC has too many NA's, must delete this also
  'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'Functional', # HouseQC
  'Neighborhood', # Location
  'Amenities1', 'Amenities2',
  'SaleCondition', 'SaleType',
  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
  'FireplaceQu', 'PavedDrive',
  'Heating', 'HeatingQC', 'Electrical',
  "RoofMatl", "Exterior1st", "Exterior2nd", 'MasVnrType', 'Exterior1st', 'Exterior2nd',
  'KitchenAbvGr', 'MasVnrArea', 'GarageFinish', 'Foundation'
)
DFfactors = DFfactors[, !(names(DFfactors) %in% dropFactVars)]


cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

# Skewness
for(i in 1:ncol(DFnumeric)){
  if (abs(skew(DFnumeric[,i]))>0.8){
    DFnumeric[,i] <- log(DFnumeric[,i] +1)
  }
}

# DFnumeric = read.csv('df_numeric.csv')
# DFfactors = read.csv('df_factors.csv')
# write.csv(DFnumeric, 'df_numeric.csv')
# write.csv(DFfactors, 'df_factors.csv')

# Normalize the data by centering and scaling
PreNum <- preProcess(DFnumeric, method = c("center", "scale"))
print(PreNum)

DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)

# Dummify the remaining factor (categorical) columns
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors)) # flattening
dim(DFdummies)

# Check to see if there are any unused labels that can be discarded
ZerocolTest <- which(colSums(DFdummies[(nrow(all[!is.na(all$SalePrice),])+1):nrow(all),]) == 0)
#colnames(DFdummies[ZerocolTest])
#DFdummies = DFdummies[, -ZerocolTest]

ZerocolTrain <- which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice),]),]) == 0)
colnames(DFdummies[ZerocolTrain])
DFdummies <- DFdummies[, -ZerocolTrain] #removing predictor

# 8 is roughly 0.5% of the total observations (1448) from train set
LowCountTrain = which(colSums(DFdummies[1:nrow(all[!is.na(all$SalePrice), ]), ]) < 8)
colnames(DFdummies[LowCountTrain])
DFdummies = DFdummies[, -LowCountTrain]

# Two-Story dummy 0 or 1
DFdummies$Two_Story_dum = ifelse(all$X2ndFlrSF > 0, 1, 0)

# BsmtQC (numeric) made of BsmtQual, BsmtCond, BsmtExposure
DFdummies$BsmtQC = (all$BsmtQual + all$BsmtCond + all$BsmtExposure)/3

# Flatroof dum
all$RoofStyle = as.character(all$RoofStyle)
DFdummies$FlatRoof_dum = ifelse(all$RoofStyle == 'Flat', 1, 0)
all$RoofStyle = as.factor(all$RoofStyle)

DFdummies = DFdummies[, names(DFdummies) %in% c('Location', 'Amenities', 'Railroad', 'LandContourLvl', 'NeighRich', 'IsNew', 'Remod', 'FlatRoof_dum', 'Two_Story_dum', 'BsmtQC', 'CentralAir', 'KitchenQual', 'GarageQual')]

# merge all features back into one dataframe, all numbers (since categories dummified)
combined <- cbind(DFnorm, DFdummies) #combining all (now numeric) predictors into one dataframe 
combined$SalePrice = all$SalePrice

### TRAIN TEST SPLIT
train <- combined[!is.na(all$SalePrice), !(names(combined) %in% c('PricePerSF'))]
test <- combined[is.na(all$SalePrice), !(names(combined) %in% c('PricePerSF'))]

qqnorm(train$SalePrice)
qqline(train$SalePrice)

train$SalePrice = log(train$SalePrice)
### REGULARIZED REGRESSION w/ Ridge, Lasso ###
# Train test split: Create an 80% - 20% train-test split
x.train = model.matrix(SalePrice ~ ., train)[ , -1]
y.train = train$SalePrice

set.seed(0)
test = test[, !(names(test) %in% c('SalePrice', 'PricePerSF'))]
x.test = model.matrix( ~ ., test)[ , -1]

# length(train)/nrow(x)  # 0.7938144
# length(y.test)/nrow(x) # 0.2061856


############################################ LASSO ###########################################

#  Fit the ridge regression. Alpha = 0 for ridge regression.
library(glmnet)
library(car)
# Use glmnet to fit a ridge regression model on training by setting up a
# grid of lambda values 10^seq(5, -2, length = 100).
grid = 10^seq(3, -5, length = 100)

#Fitting the lasso regression. Alpha = 1 for lasso regression.
set.seed(0)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = grid)

lasso_mod <- train(x=x.train, y=y.train, method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune

min(lasso_mod$results$RMSE)

lasso.models = glmnet(x.train, y.train, alpha = 1, lambda = grid)

lbs_fun <- function(lasso.models, ...) {
  L <- length(lasso.models$lambda)
  x <- log(lasso.models$lambda[L])
  y <- lasso.models$beta[, L]
  labs <- names(sort(y))
  legend('topright', legend=labs, col=1:6, lty=1) # only 6 colors
}
# Plot the coefficients of these models
plot(lasso.models, xvar="lambda", main = "Lasso Regression")
lbs_fun(lasso.models)

#The coefficients all seem to shrink towards 0 as lambda gets quite large. 
#All coefficients seem to go to 0 once the log lambda value gets to about 0. 
#Note that coefficients are necessarily set to exactly 0 for lasso regression.

cv.lasso.out = cv.glmnet(x.train, y.train, alpha = 1, nfolds = 10, lambda = grid)

#  Create a plot associated with the 10-fold cross validation
plot(cv.lasso.out, main = "Lasso Regression\n")

#7 Results
bestlambda.lasso = cv.lasso.out$lambda.min
bestlambda.lasso       # 0.003
log(bestlambda.lasso)  # -5.698

#8 Fit a model
#  Fit a lasso regression model using the best lambda on the test dataset.
lasso.bestlambdatrain = predict(lasso.models, x.test, s = bestlambda.lasso)
mean((lasso.bestlambdatrain - y.train)^2)  # 0.5060285
#The test MSE is about 0.5060 with the best lambda.

#9 Refit a model & Results
lasso.best_refit = glmnet(x.train, y.train, alpha = 1)
predict(lasso.best_refit, type = "coefficients", s = bestlambda.lasso)

model.tsf = lm(SalePrice ~ TotalSqFeet + Location + HouseQC, train)
summary(model.tsf)
influencePlot(model.tsf)

vif(model.tsf)
avPlots(model.tsf)

### importance
lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0)) # GrLivArea (0.288), HouseQC (0.186), CentralAir (0.170), NeighRich (0.161), Location (0.154), Two_Story_dum (0.151), BsmtQC (0.145), TotalSqFeet (0.12), KitchenQual (0.11)
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

model.saturated = lm(SalePrice ~ ., train)
summary(model.saturated) # R-Squared = 0.88

LassoPred <- predict(lasso_mod, x.test)
predictions_lasso <- exp(LassoPred) #need to reverse the log to the real values
head(predictions_lasso)

influencePlot(model.saturated)
vif(model.saturated)
avPlots(model.saturated)

# XGboost
xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)

#xgb_caret <- train(x=x.train, y=y.train, method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
#xgb_caret$bestTune

label_train <- y.train

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = x.train, label = label_train)
dtest <- xgb.DMatrix(data = x.test)

default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=4, #default=1
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 500, nfold = 5, showsd = T, stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)

#train the model using the best iteration found by cross validation
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 266)

XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred) #need to reverse the log to the real values
head(predictions_XGB)

#view variable importance plot
library(Ckmeans.1d.dp) #required for ggplot clustering
mat <- xgb.importance (feature_names = colnames(x.train),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = TRUE)
