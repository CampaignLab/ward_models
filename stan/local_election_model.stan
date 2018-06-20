/*
  Multi-level regression model of factors affecting local election results for
  council wards, with ward and local authority level predictors.
*/
data {
  int<lower=1> N_ward;                               // number of wards 
  int<lower=1> N_authority;                          // number of local authorities 
  int<lower=1> K_ward;                               // number of ward level predictors
  int<lower=1> K_authority;                          // number of local authority level predictors
  matrix[N_authority, K_authority] X_authority;      // authority-level predictors
  matrix[N_ward, K_ward] X_ward;                     // ward-level predictors
  int<lower=1,upper=N_authority> authority[N_ward];  // maps wards to authorities
  vector[N_ward] labour_vote;                        // response variable
}
parameters {
  vector[K_ward] b_ward;
  vector[K_authority] b_authority;
  vector[N_authority] authority_error_z;
  real mu;
  real<lower=0> sigma_ward;
  real<lower=0> sigma_authority;
}
transformed parameters {
  vector[N_authority] authority_error = authority_error_z * sigma_authority;
  vector[N_authority] authority_effect = X_authority * b_authority + authority_error;
  vector[N_ward] labour_vote_hat = mu + authority_effect[authority] + X_ward * b_ward;
}
model {
  // likelihood
  labour_vote ~ normal(labour_vote_hat, sigma_ward);
  // priors
  b_ward ~ normal(0, 1);
  b_authority ~ normal(0, 1);
  authority_error_z ~ normal(0, 1);
  sigma_authority ~ student_t(4, 0, 1);
  sigma_ward ~ student_t(4, 0, 1);
}
generated quantities {
  vector[N_ward] log_likelihood;
  vector[N_ward] labour_vote_tilde;
  for (n in 1:N_ward){
    log_likelihood[n] = normal_lpdf(labour_vote[n] | labour_vote_hat[n], sigma_ward);
    labour_vote_tilde[n] = normal_rng(labour_vote_hat[n], sigma_ward);
  }
}
