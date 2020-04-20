# Write a function that generate log-likelihood, gradient and Hessian
# Inputs: 
# t - days since first case
# y - outcome
# par - vector containing a, b, and c parameters
func = function(t, y, a, b, c) {
  
  # Expu
  expu = exp(-b * (t - c))
  
  # Log-likelihood
  loglik = sum(log(a) - log(1 + expu))
  
  # Gradient
  grad = vector(mode = "numeric", 3)
  
  grad[1] = 1/a
  grad[2] = sum((1 / (1 + expu)) * (c - t) * expu)
  grad[3] = sum((1 / (1 + expu)) * b * expu)
  
  # Hessian Matrix
  hess = matrix(0, 3, 3)
  hess[1,1] = - 1 / a^2
  hess[2,2] = sum(((1 + expu) * (-(t - c)^2 * expu) + (t - c)^2 * expu * exp(-2)) / (1 + expu)^2)
  hess[3,3] = sum(((1 + expu) * (-b^2 * expu) + b^2 * expu * exp(-2)) / (1 + expu)^2)
  
  # Remainder of hessian matrix
  # Not negative definite?
  for (i in 1:3) {
    for (j in 1:3) {
      if (i == j) {
        next()
      } else {
        hess[i,j] = grad[i] * grad[j]
      }
    }
  }
  
  return(list(loglik = loglik, grad = grad, Hess = hess)) 
}

