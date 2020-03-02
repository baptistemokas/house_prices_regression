house_train = read.csv('train.csv')
house_test = read.csv('test.csv')

# Notes
# Columns definitely drop: Alley (1369 NA's)

# Column maybe drop: 

is.na(house_train)

library(Hmisc)
hist.data.frame(house_train$LotArea)

house_train
sum(is.na(house_test))
sum(is.na(house_train))

house_train %>% 
  select(everything(), -SalePrice)


library(tidyr)
library(ggplot2)
library(dplyr)
# or `library(tidyverse)`

house_train %>% gather() %>% head()

ggplot(gather(house_train), aes(value)) + 
  geom_histogram(bins = 10, stat = 'count') + 
  facet_wrap(~key, scales = 'free_x')

my_plots <- lapply(names(house_train), function(var_x){
  p <- 
    ggplot(house_train) +
    aes_string(var_x)
  
  if(is.numeric(house_train[[var_x]])) {
    p <- p + geom_density()
  }# else {
    #p <- p + geom_bar()
  #} 
})

plot_grid(plotlist = my_plots)

library(caret)
mean(house_train$LotFrontage, na.rm = TRUE) #Mean of x2 prior to imputation.






