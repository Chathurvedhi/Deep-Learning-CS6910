# Sweep Info 

## Sweep 1:
Activations :  Tanh, Relu, sigmoid -> Decided
Optimizers : Adam, Nadam, RMSprop -> Decided
Weight_init : Xavier -> Decided
Lower Learning Rates
Lower beta1/beta values
Lesser number of hidden layers

## Sweep 2:
Actvations : Sigmoid removed
Optimizers : RMSprop removed
Higher epochs have better chances
beta2 = 0.999 when sigmoid removed
Lesser number of Hidden Layers have better Accuracy
Higher number of batch size is better
weight decay, layer size, epsilon less significance

## Sweep 3: 
Final Run with more values of learning rate, epoch and batch sizes