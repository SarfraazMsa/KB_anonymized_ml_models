setwd("~/Videos/kb_smu_child_malnutrition")
library(tidyverse)
library(broom)
df<- read.csv(file="live_still_birth_assumptions_check.csv", header=TRUE, sep=",")
predictors <- colnames(df)

model <- glm(live_still_birth ~., data = df, 
             family = binomial)


# Predict the probability (p) of diabete positivity
probabilities <- predict(model, type = "response")
predicted.classes <- df$live_still_birth

# Select only numeric predictors
mydata <- df %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")