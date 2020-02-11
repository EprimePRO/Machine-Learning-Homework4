library(HSAUR)
data("plasma")
df <- data.frame(plasma)
set.seed(1234)
glm1 <- glm(ESR~fibrinogen, family="binomial", data = df)
summary(glm1)

sigmoid <- function(z){
  1.0 / (1+exp(-z))
}

weights <- c(1,1)
data_matrix <- cbind(rep(1, nrow(df)), df$fibrinogen)
labels <- as.integer(df$ESR)
  