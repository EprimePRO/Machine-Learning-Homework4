library(HSAUR)
attach("plasma")
#data exploration
str(plasma)
head(plasma)
set.seed(1234)
glm1 <- glm(ESR~fibrinogen, family="binomial", data = plasma)
summary(glm1)
summary(plasma$ESR)
summary(plasma$fibrinogen)

#linear regression from scrat
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
par(mfrow=c(1,2))
cdplot(plasma$ESR~plasma$fibrinogen)
plot(plasma$fibrinogen, plasma_log_odds, col=plasma$ESR)
abline(weights[1], weights[2])


