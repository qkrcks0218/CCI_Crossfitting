expit <- function(v){exp(v)/(1+exp(v))}

set.seed(3)

N  <- 3000
X  <- runif(N)-0.5
PS <- expit(X)
A  <- rbinom(N,1,PS)

OR0 <- X
OR1 <- 1+3*X^3+2*X^2+X

Y0 <- OR0+rnorm(N)
Y1 <- OR1+rnorm(N)
Y  <- A*Y1+(1-A)*Y0

data <- data.frame(cbind(Y,A,X))
colnames(data) <- c("Y","A","X")
data.A1 <- data.A0 <- data
data.A1$A <- 1
data.A0$A <- 0

## CATE

CATE <- OR1-OR0

par(mfrow=c(1,3))
plot(X,OR0); plot(X,OR1); plot(X,CATE)


## Split Sample

S.Index <- sample(N,N)
EstFold <- FWFold <- EvalFold <- list()
EstFold[[1]] <- FWFold[[2]] <- EvalFold[[3]] <- S.Index[1:(N/3)]
EstFold[[2]] <- FWFold[[3]] <- EvalFold[[1]] <- S.Index[(1*N/3)+1:(N/3)]
EstFold[[3]] <- FWFold[[1]] <- EvalFold[[2]] <- S.Index[(2*N/3)+1:(N/3)]



################################################################################
# Super Learner
################################################################################

source("0.MySL.R")

SL.hpara <- list()
SL.hpara$SLL <- c(1,7) # c(1,2,3,4,5,6,7,9,10)
# Superlearner basic learning algorithms:
# 1: GLM
# 2: lasso/ridge
# 3: earth
# 4: GAM
# 5: xgboost
# 6: polynomial spline
# 7: random forest
# 9: gbm
# 10: 1-layer MLP
SL.hpara$MTRY <- c(1)                  # random forest parameters
SL.hpara$MLPL <- c(2,4)                   # number of nodes in MLP
SL.hpara$NMN <- 2                        # gbm parameter
SL.hpara$MLPdecay <- 10^c(-4,-5)          # MLP decay parameter

## Outcome Regression: Y~A+X1+X2
## Propensity score: A~X1+X2
pos.Y <- 1
pos.A <- 2
pos.X <- 3
pos.AX <- 2:3

## Estimation of Nuisance Ft

MU.Est <- PS.Est <- list()
for(ss in 1:3){
  MU.Est[[ss]] <- 
    MySL(Data=data[EstFold[[ss]],],
         locY = pos.Y,
         locX = pos.AX,
         Ydist = gaussian(),
         SL.list=SL.hpara$SLL, 
         MTRY=SL.hpara$MTRY)
  
  PS.Est[[ss]] <- 
    MySL(Data=data[EstFold[[ss]],],
         locY = pos.A,
         locX = pos.X,
         Ydist = binomial(),
         SL.list=SL.hpara$SLL, 
         MTRY=SL.hpara$MTRY)
}

## Evaluation of Nuisance Ft

MU.A0.ML <- MU.A1.ML <- PS.ML <- list()
for(ss in 1:3){
  MU.A0.ML[[ss]] <- predict(MU.Est[[ss]],
                            newdata=data.A0[FWFold[[ss]],pos.AX],
                            onlySL=TRUE)$pred
  MU.A1.ML[[ss]] <- predict(MU.Est[[ss]],
                            newdata=data.A1[FWFold[[ss]],pos.AX],
                            onlySL=TRUE)$pred
  PS.ML[[ss]] <- predict(PS.Est[[ss]],
                         newdata=data.frame(X=data[FWFold[[ss]],pos.X]),
                         onlySL=TRUE)$pred
}

CATE.A0.ML.Vec <- CATE.A1.ML.Vec <- rep(0,N)
for(ss in 1:3){
  CATE.A1.ML.Vec[FWFold[[ss]]] <- MU.A1.ML[[ss]]
  CATE.A0.ML.Vec[FWFold[[ss]]] <- MU.A0.ML[[ss]]
}

## FW: worth trying Yachong Yang's Github Code 
## https://github.com/Elsa-Yang98/Forster_Warmuth_counterfactual_regression

## Create pseudo-outcome for CATE

CATE.AIPW <- list()
for(ss in 1:3){
  
  Y.FW <- Y[FWFold[[ss]]]
  A.FW <- A[FWFold[[ss]]]
  X.FW <- X[FWFold[[ss]]]
  
  CATE.AIPW[[ss]] <- ( A.FW*(Y.FW-MU.A1.ML[[ss]])/PS.ML[[ss]] ) - 
    ( (1-A.FW)*(Y.FW-MU.A0.ML[[ss]])/(1-PS.ML[[ss]]) ) + 
    ( MU.A1.ML[[ss]] - MU.A0.ML[[ss]] )
}

J <- 4
Basis.X <- list()
for(ss in 1:3){
  Basis.X[[ss]] <- bs(X[FWFold[[ss]]],
                      degree=J,
                      intercept = T,
                      Boundary.knots = range(X))
}

FW <- function(x,Basis,Outcome){
  phix <- matrix(predict(Basis,x),J+1,1)
  INV <- solve( t(Basis)%*%(Basis) + phix%*%t(phix) )
  hx  <- as.numeric( t(phix)%*%INV%*%phix )
  numer <- as.numeric( t(phix)%*%INV%*%(t(Basis)%*%Outcome) )
  return( (1-hx)*numer )
}

CATE.FW <- list()
for(ss in 1:3){
  CATE.FW[[ss]] <- rep(0,N/3)
  for(jj in 1:(N/3)){
    CATE.FW[[ss]][jj] <- 
      FW(x = X[EvalFold[[ss]]][jj],
         Basis = Basis.X[[ss]],
         Outcome = CATE.AIPW[[ss]])
  }
}

CATE.FW.Vec <- rep(0,N)
for(ss in 1:3){
  CATE.FW.Vec[EvalFold[[ss]]] <- CATE.FW[[ss]]
}

par(mfrow=c(1,2))
plot(X,CATE.A1.ML.Vec-CATE.A0.ML.Vec,pch=19,cex=0.5)
lines(X[order(X)],CATE[order(X)],col=2,lwd=2)
plot(X,CATE.FW.Vec,pch=19,cex=0.5)
lines(X[order(X)],CATE[order(X)],col=2,lwd=2)


