# Bayesian-Networks

In this project, you will implement the following four algorithms and test their performance on 10 datasets available on the class web page.

- Independent Bayesian networks. Assume the following structure: The Bayesian networks has no edges. Learn the parameters of the independent Bayesian network using the maximum likelihood approach. Use 1-Laplace smoothing to ensure that you don’t have any zero probabilities in the model.

- Tree Bayesian networks. Use the Chow-Liu algorithm to learn the structure and parameters of the Bayesian network. Use 1-Laplace smoothing to ensure that you don’t have any zeros when computing the mutual information as well as zero probabilities in the model. See section 2 in [Meila and Jordan, 2001].

- Mixtures of Tree Bayesian networks using EM. The model is defined as follows. We have one latent variable having k values and each mixture component is a Tree Bayesian network. Learn the structure and parameters of the model using the EM-algorithm (in the M-step each mixture component is learned using the Chow-Liu algorithm). Select k using the validation set and use 1 - Laplace smoothing. Run the EM algorithm until convergence or until 100 iterations whichever is earlier. See section 3 in [Meila and Jordan, 2001].

- Mixtures of Tree Bayesian networks using Random Forests. The model is defined as above (see Item (3)). Learn the structure and parameters of the model using the following Random-Forests style approach. Given two hyper-parameters (k, r), generate k sets of Bootstrap samples and learn the i-th Tree Bayesian network 1 using the i-th set of the Bootstrap samples by randomly setting exactly r mutual information scores to 0 (as before use the Chow-Liu algorithm with r mutual information scores set to 0 to learn the structure and parameters of the Tree Bayesian network). Select k and r using the validation set and use 1-Laplace smoothing. You can either set pi = 1/k for all i or use any reasonable method (reasonable method is extra credit). Describe your (reasonable) method precisely in your report. Does it improve over the baseline approach that uses pi = 1/k.

# Conclusion

Mixture of Tree Bayesian Networks using EM gives the best performance followed by Tree Bayesian Networks and finally by Tree Bayesian Networks using Random Forest. This might be attributed to the ability of TBNs using EM that they can capture more complex relationships between the input features and the output variable, which might not be captured by a single TBN or TBN-RF.
