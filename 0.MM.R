library(MASS)

WS <- function(x,lb,ub){
  if(is.null(x)){
    RR <- median(x,lb,ub)
  } else {
    RR <- apply(cbind(x,lb,ub),1,median)
  }
  return(RR)
}

expit <- function(x){
  exp(x)/(1+exp(x))
}

MM <- function(X.Input){
  if( is.null(dim(X.Input)) ){
    X.Output <- matrix(X.Input,length(X.Input),1)
  } else {
    X.Output <- matrix(0,dim(X.Input)[1],dim(X.Input)[2])
    for(tt in 1:dim(X.Input)[2]){
      X.Output[,tt] <- as.numeric( X.Input[,tt] )
    }
  }
  return(X.Output)
}

FT_RBF <- function(X,X.new=F,bw=1){
  
  X <- MM(X)
  X.new <- MM(X.new)
  DM <- matrix(0,dim(X.new)[1],dim(X)[1])
  for(ii in 1:dim(X.new)[1]){
    for(jj in 1:dim(X)[1]){
      DM[ii,jj] <- exp(-sum((X.new[ii,]-X[jj,])^2)/bw)
    }
  }
  return(DM)
}

FT_BWHeuristic <- function(X,p=0.5){
  X <- MM(X)
  c( quantile( c(dist(X))^2, p ) )
}

FT_Minimax <- function(Coef,
                       Intercept,
                       X.Perturb,
                       X.Target,
                       bw.Perturb=1,
                       bw.Target=1,
                       lambda.Perturb=1,
                       lambda.Target=1,
                       SVM.bias=F,
                       bias.input=NULL){
  
  N <- length(Coef)
  
  X.Perturb <- MM(X.Perturb) ; X.Target <- MM(X.Target)
  Kmat.Perturb <- FT_RBF(X.Perturb,
                         X.new=X.Perturb,
                         bw.Perturb)
  Kmat.Target <- FT_RBF(X.Target,
                        X.new=X.Target,
                        bw.Target)
  
  if(SVM.bias){
    if(is.null(bias.input)){
      SVM.BIAS <- -mean( Intercept )/mean( Coef )
    } else {
      SVM.BIAS <- bias.input
    }
    Intercept.Original <- Intercept
    Intercept <- Intercept + SVM.BIAS*Coef
    
    Gamma <- try( 0.25*Kmat.Perturb%*%ginv(Kmat.Perturb/N + diag(rep(lambda.Perturb,N))), silent=TRUE )
    if(class(Gamma)[1]=="try-error"){
      
      RESULT <- list()
      RESULT$gamma <- rep(0,N)
      RESULT$bias  <- SVM.BIAS
      
      return(RESULT)
      
    }
    D <- diag(as.vector(Coef))
    v <- matrix(Intercept,N,1)
    
    gamma <- try( -ginv(Kmat.Target%*%D%*%Gamma%*%D%*%Kmat.Target+N^2*lambda.Target*Kmat.Target)%*%(Kmat.Target%*%D%*%Gamma%*%v), 
                  silent=TRUE )
    if(class(gamma)[1]=="try-error"){
      RESULT <- list()
      RESULT$gamma <- rep(0,N)
      RESULT$bias  <- SVM.BIAS
      
      return(RESULT)
    }
    
    RESULT <- list()
    RESULT$gamma <- gamma
    RESULT$bias  <- SVM.BIAS
    
    return(RESULT)
    
  } else {
    
    SVM.BIAS <- 0
    
    Gamma <- try( 0.25*Kmat.Perturb%*%ginv(Kmat.Perturb/N + diag(rep(lambda.Perturb,N))), silent=TRUE )
    if(class(Gamma)[1]=="try-error"){
      
      RESULT <- list()
      RESULT$gamma <- rep(0,N)
      RESULT$bias  <- SVM.BIAS
      
      return(RESULT)
      
    }
    D <- diag(as.vector(Coef))
    v <- matrix(Intercept,N,1)
    
    gamma <- try( -ginv(Kmat.Target%*%D%*%Gamma%*%D%*%Kmat.Target+N^2*lambda.Target*Kmat.Target)%*%
                    (Kmat.Target%*%D%*%Gamma%*%v),
                  silent=TRUE )
    if(class(gamma)[1]=="try-error"){
      RESULT <- list()
      RESULT$gamma <- rep(0,N)
      RESULT$bias  <- SVM.BIAS
      
      return(RESULT)
    }
    
    
    RESULT <- list()
    RESULT$gamma <- gamma
    RESULT$bias  <- SVM.BIAS
    
    return(RESULT)
    
  }
  
  
}

FT_CV_Minimax <- function( Coef.Train,
                           Intercept.Train,
                           X.Perturb.Train,
                           X.Target.Train,
                           Coef.Valid,
                           Intercept.Valid,
                           X.Perturb.Valid,
                           X.Target.Valid,
                           bw.Perturb=1,
                           bw.Target=1,
                           lambda.Perturb=1,
                           lambda.Target=1,
                           bound=c(0.01,100),
                           SVM.bias=F,
                           bias.input=NULL){
  
  X.Perturb.Train <- MM(X.Perturb.Train)
  X.Target.Train  <- MM(X.Target.Train)
  
  X.Perturb.Valid <- MM(X.Perturb.Valid)
  X.Target.Valid  <- MM(X.Target.Valid)
  
  RKHS.Coef <- FT_Minimax( Coef = Coef.Train,
                           Intercept = Intercept.Train,
                           X.Perturb = X.Perturb.Train,
                           X.Target  = X.Target.Train,
                           bw.Perturb,
                           bw.Target,
                           lambda.Perturb,
                           lambda.Target,
                           SVM.bias,
                           bias.input) 
  
  K.Target.Test <- FT_RBF(X=X.Target.Train,
                          X.new=X.Target.Valid,
                          bw.Target)
  
  Kmat.Target <- FT_RBF(X.Target.Valid,
                        X.new=X.Target.Valid,
                        bw.Target)
  
  Kmat.Perturb <- FT_RBF(X.Perturb.Valid,
                         X.new=X.Perturb.Valid,
                         bw.Perturb)
  
  Beta.Original <- (K.Target.Test%*%RKHS.Coef$gamma + RKHS.Coef$bias)
  Beta <- WS( Beta.Original, bound[1], bound[2] )
  Error.Add <- sum(as.numeric((Beta-Beta.Original)^2))
  
  N.Test <- length(Coef.Valid)
  
  xi <- (Intercept.Valid + Beta*Coef.Valid)/N.Test
  
  Gamma <- try( 0.25*Kmat.Perturb%*%ginv(Kmat.Perturb/N.Test + diag(rep(lambda.Perturb,N.Test))),silent=TRUE )
  if(class(Gamma)[1]=="try-error"){
    Gamma <- diag(rep(0.25*N.Test,N.Test)) + 0.25*Kmat.Perturb*lambda.Perturb/(1+lambda.Perturb)
  }
  
  Error.Proj <- t(xi)%*%Gamma%*%(xi) + Error.Add*100000
  
  
  Gamma <- try( 0.25*Kmat.Perturb%*%ginv(Kmat.Perturb/N.Test + diag(rep(0*lambda.Perturb,N.Test))), silent=TRUE )
  if(class(Gamma)[1]=="try-error"){
    Gamma <- diag(rep(0.25*N.Test,N.Test))
  }
  
  Error.Proj.0 <- t(xi)%*%Gamma%*%(xi) + Error.Add*100000
  
  Error.Sq <- (sum(xi))^2 + Error.Add*100000
  
  Error.PMMR <- t(xi)%*%Kmat.Perturb%*%(xi)
  
  RESULT <- list()
  RESULT$gamma <- RKHS.Coef$gamma
  RESULT$bias <- RKHS.Coef$bias
  RESULT$estimate <- Beta
  RESULT$Error.Proj <- Error.Proj
  RESULT$Error.Proj.0 <- Error.Proj.0
  RESULT$Error.Sq <- Error.Sq
  RESULT$Error.PMMR <- Error.PMMR
  
  return(RESULT)
}







