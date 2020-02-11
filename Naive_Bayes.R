df <- read.csv("titanic_project.csv")

#make survived a factor
df$survived <- as.factor(df$survived)

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
i <- sample(1:nrow(df), 900, replace = FALSE)
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
