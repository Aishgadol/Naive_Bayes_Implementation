import naive_bayes_implementation

condDict,priorDict=naive_bayes_implementation.naiveBayes_Learn()
naive_bayes_implementation.naiveBayes_Test(condDict,priorDict)