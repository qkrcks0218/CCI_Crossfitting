expit <- function(v){exp(v)/(1+exp(v))}

set.seed(2)

N  <- 500
X1 <- rnorm(N)
X2 <- rnorm(N)

PS <- expit((X1+X2)/4)
A  <- rbinom(N,1,PS)

OR0 <- X1+X2
OR1 <- 1+0.5*X1+2*X2

Y0 <- OR0+rnorm(N)
Y1 <- OR1+rnorm(N)
Y  <- A*Y1+(1-A)*Y0

data <- data.frame(cbind(Y,A,X1,X2))
colnames(data) <- c("Y","A","X1","X2")
data.A1 <- data.A0 <- data
data.A1$A <- 1
data.A0$A <- 0

## ATE

tau <- 1; tau1 <- 1 #E[Y(1)-Y(0)]; E[Y(10)]

## Split Sample

S.Index <- sample(N,N)
EstFold <- EvalFold <- list()
EstFold[[1]] <- EvalFold[[2]] <- S.Index[1:(N/2)]
EstFold[[2]] <- EvalFold[[1]] <- S.Index[(N/2)+1:(N/2)]

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
SL.hpara$MTRY <- c(1,2)                  # random forest parameters
SL.hpara$MLPL <- c(2,4)                   # number of nodes in MLP
SL.hpara$NMN <- 2                        # gbm parameter
SL.hpara$MLPdecay <- 10^c(-4,-5)          # MLP decay parameter

## Outcome Regression: Y~A+X1+X2
## Propensity score: A~X1+X2
pos.Y <- 1
pos.A <- 2
pos.X <- 3:4
pos.AX <- 2:4

## Estimation of Nuisance Ft

MU.Est <- PS.Est <- list()
for(ss in 1:2){
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

MU.A0.Eval <- MU.A1.Eval <- PS.Eval <- list()
for(ss in 1:2){
  MU.A0.Eval[[ss]] <- predict(MU.Est[[ss]],
                              newdata=data.A0[EvalFold[[ss]],pos.AX],
                              onlySL=TRUE)$pred
  MU.A1.Eval[[ss]] <- predict(MU.Est[[ss]],
                              newdata=data.A1[EvalFold[[ss]],pos.AX],
                              onlySL=TRUE)$pred
  PS.Eval[[ss]] <- predict(PS.Est[[ss]],
                           newdata=data[EvalFold[[ss]],pos.X],
                           onlySL=TRUE)$pred
}

MU.A0.Eval.Vec <- MU.A1.Eval.Vec <- PS.Eval.Vec <- rep(0,N)
for(ss in 1:2){
  MU.A0.Eval.Vec[EvalFold[[ss]]] <- MU.A0.Eval[[ss]]
  MU.A1.Eval.Vec[EvalFold[[ss]]] <- MU.A1.Eval[[ss]]
  PS.Eval.Vec[EvalFold[[ss]]] <- PS.Eval[[ss]]
}

## Visual check

par(mfrow=c(1,3))
plot(PS, PS.Eval.Vec); abline(a=0,b=1,col=2)
plot(OR0,MU.A0.Eval.Vec); abline(a=0,b=1,col=2)
plot(OR1,MU.A1.Eval.Vec); abline(a=0,b=1,col=2)

## Influence function & Estimate

InfFt <- ( A*(Y-MU.A1.Eval.Vec)/PS.Eval.Vec ) - 
  ( (1-A)*(Y-MU.A0.Eval.Vec)/(1-PS.Eval.Vec) ) + 
  ( MU.A1.Eval.Vec - MU.A0.Eval.Vec )
mean(InfFt); tau


################################################################################
# Minimax
################################################################################

source("0.MM.R")

X <- cbind(X1,X2)
bX <- as.numeric( FT_BWHeuristic(X) )

ParaGrid.nu <- 
  expand.grid(bX,
              bX,
              10^c(-4,-2),
              10^c(-4,-2))

ParaGrid.pi <- 
  expand.grid(bX,
              bX,
              10^c(-4,-2),
              10^c(-4,-2))
CF    <- 2
NumCV <- 5
Bound.nu  <- c(-100,100)
Bound.pi  <- c(0,100)

CV.Index <- list()
CV.Index[[1]] <- CV.Index[[2]] <- list()
CV.Index[[1]][[1]] <- 0*50+1:50
CV.Index[[1]][[2]] <- 1*50+1:50
CV.Index[[1]][[3]] <- 2*50+1:50
CV.Index[[1]][[4]] <- 3*50+1:50
CV.Index[[1]][[5]] <- 4*50+1:50

CV.Index[[2]][[1]] <- 0*50+1:50
CV.Index[[2]][[2]] <- 1*50+1:50
CV.Index[[2]][[3]] <- 2*50+1:50
CV.Index[[2]][[4]] <- 3*50+1:50
CV.Index[[2]][[5]] <- 4*50+1:50

RISK.nu <- RISK.pi <- list() 
for(ss in 1:CF){
  RISK.nu[[ss]] <- RISK.pi[[ss]] <- matrix(10^5,dim(ParaGrid.nu)[1],4*NumCV) 
  
  colnames(RISK.nu[[ss]]) <- colnames(RISK.pi[[ss]]) <-    
    c(sprintf("Pr%0.1d",1:NumCV),
      sprintf("P0%0.1d",1:NumCV),
      sprintf("Sq%0.1d",1:NumCV),
      sprintf("PMMR%0.1d",1:NumCV))
}

## CV

for(Para.Iter in 1:dim(ParaGrid.nu)[1]){
  
  for(ss in 1:CF){
    for(cv in 1:NumCV){
      
      CV.Split.Index <- list()
      CV.Split.Index[[1]] <- EstFold[[ss]][ -CV.Index[[ss]][[cv]] ]
      CV.Split.Index[[2]] <- EstFold[[ss]][ CV.Index[[ss]][[cv]] ]
      
      Y.CV  <- list()
      A.CV  <- list()
      X.CV  <- list() 
      
      ## \nu
      
      nuhat.ML.CV <- list()
      
      for(cvest in 1:2){
        Y.CV [[cvest]] <- Y[CV.Split.Index[[cvest]] ]
        A.CV [[cvest]] <- A[CV.Split.Index[[cvest]] ]
        X.CV [[cvest]] <- X[CV.Split.Index[[cvest]],]
      }
      
      ## nu
      
      pos.nu.CV <- list()
      Coef.nu.CV <- Intercept.nu.CV <- list()
      
      for(cvest in 1:2){
        pos.nu.CV[[cvest]]  <- 1:length(Y.CV[[cvest]])
        Coef.nu.CV[[cvest]] <- -A.CV[[cvest]]
        Intercept.nu.CV[[cvest]] <- A.CV[[cvest]]*Y.CV[[cvest]]
      }
      
      CV.nu <- 
        FT_CV_Minimax( Coef.Train      = Coef.nu.CV[[1]][pos.nu.CV[[1]] ],
                       Intercept.Train = Intercept.nu.CV[[1]][pos.nu.CV[[1]] ],
                       X.Perturb.Train = X.CV[[1]][pos.nu.CV[[1]],],
                       X.Target.Train  = X.CV[[1]][pos.nu.CV[[1]],],
                       Coef.Valid      = Coef.nu.CV[[2]][pos.nu.CV[[2]] ],
                       Intercept.Valid = Intercept.nu.CV[[2]][pos.nu.CV[[2]] ],
                       X.Perturb.Valid = X.CV[[2]][pos.nu.CV[[2]],],
                       X.Target.Valid  = X.CV[[2]][pos.nu.CV[[2]],],
                       bw.Perturb      = ParaGrid.nu[Para.Iter,1],
                       bw.Target       = ParaGrid.nu[Para.Iter,2],
                       lambda.Perturb  = ParaGrid.nu[Para.Iter,3],
                       lambda.Target   = ParaGrid.nu[Para.Iter,4],
                       bound = Bound.nu,
                       SVM.bias = T,
                       bias.input = mean(Y[A==1]))
      
      RISK.nu[[ss]][Para.Iter,c(cv,cv+NumCV,cv+2*NumCV,cv+3*NumCV)] <- 
        c(CV.nu$Error.Proj,
          CV.nu$Error.Proj.0,
          CV.nu$Error.Sq,
          CV.nu$Error.PMMR)
    }
    
    
    print(c(Para.Iter,ss))
  }
  print(c(Para.Iter))
}


for(Para.Iter in 1:dim(ParaGrid.nu)[1]){
  
  for(ss in 1:CF){
    for(cv in 1:NumCV){
      
      CV.Split.Index <- list()
      CV.Split.Index[[1]] <- EstFold[[ss]][ -CV.Index[[ss]][[cv]] ]
      CV.Split.Index[[2]] <- EstFold[[ss]][ CV.Index[[ss]][[cv]] ]
      
      Y.CV  <- list()
      A.CV  <- list()
      X.CV  <- list()
      
      ## \pi
      
      pihat.ML.CV <- list()
      
      for(cvest in 1:2){
        Y.CV [[cvest]] <- Y[CV.Split.Index[[cvest]] ]
        A.CV [[cvest]] <- A[CV.Split.Index[[cvest]] ]
        X.CV [[cvest]] <- X[CV.Split.Index[[cvest]],]
      }
      
      ## pi
      
      pos.pi.CV <- list()
      Coef.pi.CV <- Intercept.pi.CV <- list()
      
      for(cvest in 1:2){
        pos.pi.CV[[cvest]]  <- 1:length(Y.CV[[cvest]])
        Coef.pi.CV[[cvest]] <- A.CV[[cvest]]
        Intercept.pi.CV[[cvest]] <- rep(-1,length(Y.CV[[cvest]]))
      }
      
      CV.pi <- 
        FT_CV_Minimax( Coef.Train      = Coef.pi.CV[[1]][pos.pi.CV[[1]] ],
                       Intercept.Train = Intercept.pi.CV[[1]][pos.pi.CV[[1]] ],
                       X.Perturb.Train = X.CV[[1]][pos.pi.CV[[1]],],
                       X.Target.Train  = X.CV[[1]][pos.pi.CV[[1]],],
                       Coef.Valid      = Coef.pi.CV[[2]][pos.pi.CV[[2]] ],
                       Intercept.Valid = Intercept.pi.CV[[2]][pos.pi.CV[[2]] ],
                       X.Perturb.Valid = X.CV[[2]][pos.pi.CV[[2]],],
                       X.Target.Valid  = X.CV[[2]][pos.pi.CV[[2]],],
                       bw.Perturb      = ParaGrid.pi[Para.Iter,1],
                       bw.Target       = ParaGrid.pi[Para.Iter,2],
                       lambda.Perturb  = ParaGrid.pi[Para.Iter,3],
                       lambda.Target   = ParaGrid.pi[Para.Iter,4],
                       bound = Bound.pi,
                       SVM.bias = T,
                       bias.input = 1/mean(A))
      
      RISK.pi[[ss]][Para.Iter,c(cv,cv+NumCV,cv+2*NumCV,cv+3*NumCV)] <- 
        c(CV.pi$Error.Proj,
          CV.pi$Error.Proj.0,
          CV.pi$Error.Sq,
          CV.pi$Error.PMMR) 
    }
    
    
    print(c(Para.Iter,ss))
  }
  print(c(Para.Iter))
}



Opt.Para.nu <- Opt.Para.pi <- list()
for(ss in 1:CF){
  pos.p0 <- which( substr(colnames(RISK.nu[[ss]]),1,4)=="PMMR" )
  Opt.Para.nu[[ss]] <- 
    as.numeric(ParaGrid.nu[which.min(apply(RISK.nu[[ss]][,pos.p0],1,mean)),])
  Opt.Para.pi[[ss]] <- 
    as.numeric(ParaGrid.pi[which.min(apply(RISK.pi[[ss]][,pos.p0],1,mean)),])
}

## Estimation

Coef.nu.MM <- Intercept.nu.MM <- nu.MM <- nu.predict <- list()
Coef.pi.MM <- Intercept.pi.MM <- pi.MM <- pi.predict <- list()

for(ss in 1:CF){
  
  ## nu
  
  nu.MM[[ss]] <- 
    FT_Minimax( Coef            = -A[EstFold[[ss]]],
                Intercept       = Y[EstFold[[ss]]]*A[EstFold[[ss]]],
                X.Perturb       = X[EstFold[[ss]],],
                X.Target        = X[EstFold[[ss]],],
                bw.Perturb      = Opt.Para.nu[[ss]][1],
                bw.Target       = Opt.Para.nu[[ss]][2],
                lambda.Perturb  = Opt.Para.nu[[ss]][3],
                lambda.Target   = Opt.Para.nu[[ss]][4],
                SVM.bias = T,
                bias.input = mean(Y[A==1]))
  
  nu.predict[[ss]] <- function(X.New.Input){
    
    RKHS.Fit <- 
      (FT_RBF(X     = X[EstFold[[ss]],],
              X.new = X.New.Input,
              bw    = Opt.Para.nu[[ss]][2])%*%nu.MM[[ss]]$gamma + nu.MM[[ss]]$bias)
    
    RKHS.Fit
  }
  
} 

for(ss in 1:CF){
  
  ## pi
  
  pi.MM[[ss]] <- 
    FT_Minimax( Coef            = A[EstFold[[ss]]],
                Intercept       = rep(-1,length(EstFold[[ss]])),
                X.Perturb       = X[EstFold[[ss]],],
                X.Target        = X[EstFold[[ss]],],
                bw.Perturb      = Opt.Para.pi[[ss]][1],
                bw.Target       = Opt.Para.pi[[ss]][2],
                lambda.Perturb  = Opt.Para.pi[[ss]][3],
                lambda.Target   = Opt.Para.pi[[ss]][4],
                SVM.bias = T,
                bias.input = 1/mean(A))
  
  pi.predict[[ss]] <- function(X.New.Input){
    
    RKHS.Fit <- 
      (FT_RBF(X     = X[EstFold[[ss]],],
              X.new = X.New.Input,
              bw    = Opt.Para.pi[[ss]][2])%*%pi.MM[[ss]]$gamma + pi.MM[[ss]]$bias)
    
    RKHS.Fit
  }
}

## Evaluation

nuhat.Eval <- rep(0,N)
for(ss in 1:CF){
  nuhat.Eval[EvalFold[[ss]]] <- nu.predict[[ss]](X[EvalFold[[ss]],])
}

pihat.Eval <- rep(0,N)
for(ss in 1:CF){
  pihat.Eval[EvalFold[[ss]]] <- pi.predict[[ss]](X[EvalFold[[ss]],])
}

par(mfrow=c(1,2))
plot(OR1,nuhat.Eval); abline(a=0,b=1,col=2)
plot(1/PS,pihat.Eval); abline(a=0,b=1,col=2)

## Influence function & Estimate

InfFt1 <- ( A*pihat.Eval*(Y-nuhat.Eval) ) + nuhat.Eval
mean(InfFt1); tau1




################################################################################
# Multiplier Bootstrap
################################################################################

## ATE
mean(InfFt); tau
sqrt( var(InfFt-tau)/N )

B <- 10000
MultBoot <- rep(0,B)
for(bb in 1:B){
  MultBoot[bb] <- mean( rnorm(N)*(InfFt-tau) )
}

sd(MultBoot)


## ATE1
mean(InfFt1); tau1
sqrt( var(InfFt1-tau1)/N )

B <- 10000
MultBoot1 <- rep(0,B)
for(bb in 1:B){
  MultBoot1[bb] <- mean( rnorm(N)*(InfFt1-tau1) )
}

sd(MultBoot1)























