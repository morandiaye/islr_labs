library(MASS)
library(ISLR)

# use `Boston` data set
names(Boston)

?Boston

# fit a siple linear model with `medv` (median value of owner occupied homes) as Y, and `lstat` as predictor.
lm.fit <- lm(medv~lstat, data = Boston)
lm.fit

summary(lm.fit)
names(lm.fit)

# obtain a confidence interval