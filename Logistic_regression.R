library(HSAUR)
attach("plasma")

#data exploration
str(plasma)
head(plasma)

#set seed to get the same results in each run
set.seed(1234)

#logistic regression
glm1 <- glm(ESR~fibrinogen, family="binomial", data = plasma)

#summary of the logistic regression model
summary(glm1)

#summary of the target (Qualitative)
summary(plasma$ESR)

#summary of the predictor (Qualitative/Quantitative)
summary(plasma$fibrinogen)

#linear regression from scratch with process time
ptm <- proc.time()
sigmoid <- function(z){
  1.0 / (1+exp(-z))
}

weights <- c(1,1)
data_matrix <- cbind(rep(1, nrow(plasma)), plasma$fibrinogen)
labels <- as.integer(plasma$ESR) - 1

weights <- c(1,1)

#Gradient descent
learning_rate <- 0.001
for(i in 1:500000){
  prob_vector <- sigmoid(data_matrix %*% weights)
  error <- labels - prob_vector
  weights <- weights + learning_rate * t(data_matrix) %*% error
}
weights

#log odds
plasma_log_odds <- cbind(rep(1,32), plasma$fibrinogen) %*% weights

#print out the process time of the model
proc.time() - ptm 

#display 2 graphs cdplot and regular plot
par(mfrow=c(1,2))
cdplot(plasma$ESR~plasma$fibrinogen)
plot(plasma$fibrinogen, plasma_log_odds, col=plasma$ESR)
abline(weights[1], weights[2])

