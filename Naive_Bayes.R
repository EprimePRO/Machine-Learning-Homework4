df <- read.csv("titanic_project.csv")

#make survived a factor
df$survived <- as.factor(df$survived)
df$pclass <- as.factor(df$pclass)
df$sex <- as.factor(df$sex)


#load naive bayes
library(e1071)
set.seed(1234)

#plot data
par(mfrow=c(1,2))
cdplot(survived~sex, data = df)
cdplot(survived~pclass, data = df)

#explore data
names(df)
head(df)
tail(df)
summary(df)


#Split into train and test
i <- c(1:900)
train <- df[i,]
test <- df[-i,]

#run naive Bayes and print results
nb1 <- naiveBayes(survived~pclass+sex+age, data=train)
nb1

#test on the test data
pred <- predict(nb1, newdata = test, type = "class")
acc <- mean(pred==test$survived)
print(paste("Naive Bayes accuracy = ", acc))

#print metrics using confusionMatrix
library(caret)
confusionMatrix(pred, test$survived)

startTime <- proc.time()

df <- train
#run naive bayes from scratch
#priors
apriori <- c(
  nrow(df[df$survived=="0",])/nrow(df),
  nrow(df[df$survived=="1",])/nrow(df)
)

#get likelihoods

# get survived counts for no and yes
count_survived <- c(
  length(df$survived[df$survived=="0"]),
  length(df$survived[df$survived=="1"])
)
# likelihood for pclass
lh_pclass <- matrix(rep(0,6), ncol=3)
for (sv in c("0", "1")){
  for (pc in c("1","2","3")) {
    lh_pclass[as.integer(sv)+1, as.integer(pc)] <- 
      nrow(df[df$pclass==pc & df$survived==sv,]) / count_survived[as.integer(sv)+1]
  }
}

# likelihood for sex
lh_sex <- matrix(rep(0,4), ncol=2)
for (sv in c("0", "1")){
  for (sx in c(1, 2)) {
    lh_sex[as.integer(sv)+1, sx] <- 
      nrow(df[as.integer(df$sex)==sx & df$survived==sv,]) /
      count_survived[as.integer(sv)+1]
  }
}

#quantitative data

age_mean <- c(0, 0)
age_var <- c(0, 0)
for (sv in c("0", "1")){
  age_mean[as.integer(sv)+1] <- 
    mean(df$age[df$survived==sv])
  age_var[as.integer(sv)+1] <- 
    var(df$age[df$survived==sv])
}

#probability density

calc_age_lh <- function(v, mean_v, var_v){
  # run like this: calc_age_lh(6, 25.9, 138)
  1 / sqrt(2 * pi * var_v) * exp(-((v-mean_v)^2)/(2 * var_v))
}

#function to calculate raw probabilities

calc_raw_prob <- function(pclass, sex, age) {
  # pclass=1,2,3  sex=1,2   age=numeric
  num_s <- lh_pclass[2, pclass] * lh_sex[2, sex] * apriori[2] *
    calc_age_lh(age, age_mean[2], age_var[2])
  num_p <- lh_pclass[1, pclass] * lh_sex[1, sex] * apriori[1] *
    calc_age_lh(age, age_mean[1], age_var[1])
  denominator <- lh_pclass[2, pclass]  * lh_sex[2, sex] * calc_age_lh(age, age_mean[2], age_var[2]) * apriori[2] + 
    lh_pclass[1, pclass]  * lh_sex[1, sex] * calc_age_lh(age, age_mean[1], age_var[1]) * apriori[1]
  return (list(prob_survived <- num_s / denominator, prob_perished <- num_p / denominator))
}


#get predicions from 5 test observations
for (i in 1:nrow(test)){
  raw <- calc_raw_prob(test$pclass[i], test$sex[i], test$age[i])
  print(paste(raw[2], raw[1]))
}

endTime <- proc.time()
#elapsed Time
endTime-startTime













