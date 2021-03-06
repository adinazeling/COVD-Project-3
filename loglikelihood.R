# Write a function that generates loss, gradient and Hessian
# Inputs: 
# t - days since first case
# y - outcome
# par - vector containing a, b, and c parameters
func = function(t, y, par) {
  
  a = par[1]
  b = par[2]
  c = par[3]
  
  # Expu
  expu = exp(-b * (t - c))
  expu[expu == Inf] = 100000000
  
  # Loss function
  loss = (1/2) * sum(y - (a / (1 + expu)))^2
  
  # First derivative matrix
  p = (expu / (1 + expu))
  
  d1loss = vector(mode = "list")
  d1loss[[1]] = (1 / (1 + expu))
  d1loss[[2]] = a * (c - t) * (1 / (1 + expu)) * p
  d1loss[[3]] = a * b * (1 / (1 + expu)) * p
  
  # Gradient
  grad = vector(mode = "numeric", length = 3)
  
<<<<<<< HEAD
  grad[[1]] = -sum(y - (a / (1 + expu)) * d1loss[[1]])
  grad[[2]] = sum(y - (a / (1 + expu)) * d1loss[[2]])
  grad[[3]] = sum(y - (a / (1 + expu)) * d1loss[[3]])
=======
  grad[[1]] = -sum((y - (a / (1 + expu)) * d1loss[[1]]))
  grad[[2]] = -sum((y - (a / (1 + expu)) * d1loss[[2]]))
  grad[[3]] = -sum((y - (a / (1 + expu)) * d1loss[[3]]))
>>>>>>> fea1b915f0c384de764ef0b125633724dd1f2f3c
  
  # Second derivative matrix
  d2loss = matrix(0, 3, 3)
  d2loss[1,1] = 0
  d2loss[2,2] = sum(y - (a / (1 + expu)) %*% t(a * ((2 * (c - t)^2 * expu * exp(-2)) / (1 + expu)^3 - ((c - t)^2 * expu) / (1 + expu)^2)))
  d2loss[3,3] = sum(y - (a / (1 + expu)) %*% t(a * ((2 * b^2 * expu * exp(-2)) / (1 + expu)^3 - (b^2 * expu) / (1 + expu)^2)))
  d2loss[1,2] = sum(y - (a / (1 + expu)) %*% t((-(c - t) * expu) / (expu + 1)^2))
  d2loss[1,3] = sum(y - (a / (1 + expu)) %*% t((-b * expu) / (expu + 1)^2))
  d2loss[2,3] = sum(y - (a / (1 + expu)) %*% t((a * b * (c - t) * expu) / (expu + 1)^2 - (2 * a * b * (c - t) * expu * exp(-2)) / (expu + 1)^3 + (a * expu) / (expu + 1)^2))
  d2loss[2,1] = d2loss[1,2]
  d2loss[3,1] = d2loss[1,3]
  d2loss[3,2] = d2loss[2,3]  
  
  # Hessian
  hess = matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      hess[i,j] = sum(matrix(d1loss[[i]]) %*% t(matrix(d1loss[[j]]))) - d2loss[i,j]
    }
  }
  
  # Information Matrix
  info = matrix(0, 3, 3)
  for (i in 1:3) {
    for (j in 1:3) {
      info[i,j] = sum(matrix(d1loss[[i]]) %*% t(matrix(d1loss[[j]])))
    }
  }
  
  
  return(list(loss = -loss, grad = grad, Hess = hess, identity = diag(3), info = info)) 
}

