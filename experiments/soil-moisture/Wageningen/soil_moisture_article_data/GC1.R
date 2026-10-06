# Download data
data <- read.table("http://www.jmulti.de/download/datasets/e1.dat", skip = 6, header = TRUE)

# Only use the first 76 observations so that there are 73 observations
# left for the estimated VAR(2) model after taking first differences.
data <- data[1:76, ]

# Convert to time series object
data <- ts(data, start = c(1960, 1), frequency = 4)

# Take logs and differences
data <- diff(log(data))

# Plot data
plot(data,  main = "Dataset E1 from Lütkepohl (2007)")

# Load package
library(vars)
library(mlVAR)

# Simulate data
numlags=1
numvars = 5
samplesize = 100
pars=diag(0.8,numvars)
pars[1,2] = 0.4
pars[3,5] = 0.4
pars[2,1] = 0.1

init = matrix(1, numlags, numvars)
burnin = 10000

data = simulateVAR(pars,  means = 0, lags = numlags, Nt = samplesize, init, residuals = 0.1,
            burnin)
#colnames(data) <- c("1st", "2nd", "3th", "4th", "5th")

# Estimate model
model <- VAR(data, p = 2, type = "const")

# Look at summary statistics
summary(model)

feir <- irf(model, impulse = "V3", response = "V5",
            n.ahead = 8, ortho = FALSE, runs = 1000)

plot(feir)

# Calculate summary statistics
model_summary <- summary(model)

# Obtain variance-covariance matrix
model_summary$covres

t(chol(model_summary$covres))

oir <- irf(model, impulse = "income", response = "cons",
           n.ahead = 8, ortho = TRUE, runs = 1000, seed = 12345)

plot(oir)



