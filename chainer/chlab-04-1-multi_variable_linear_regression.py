#!/usr/bin/env python
# Lab 2-1 Linear Regression

import numpy as np
import chainer
from chainer import training
from chainer import datasets
from chainer.training import extensions

import chainer.functions as F
import chainer.links as L


class MyModel(chainer.Chain):
    # Define model to be called later by L.Classifier()

    def __init__(self, n_in, n_out):
        super(MyModel, self).__init__(
            l1=L.Linear(n_in, n_out),
        )

    def __call__(self, x):
        return self.l1(x)


def generate_data(n_in):
    # Need to reshape so that each input is an array.
    reshape = lambda x: np.reshape(x, (len(x), n_in))

    # Notice the type specification (np.float32)
    # For regression, use np.float32 for both input & output, while for
    # classification using softmax_cross_entropy, the output(label) needs to be
    # of type np.int32.
    W = np.random.normal(5,2,n_in)
    print("set W is:", W)
    X = np.random.normal(0.5, 0.1, (100, n_in)).astype(np.float32)
    Y = (X.dot(W) + np.random.randn(X.shape[0]) * 0.33).astype(np.float32).reshape((len(X), 1))
    return reshape(X), Y


def main():
    epoch = 100
    batch_size = 1
    n_in = 4

    data = generate_data(n_in)

    # Convert to set of tuples (target, label).
    train = datasets.TupleDataset(*data)

    model = L.Classifier(MyModel(4, 1), lossfun=F.mean_squared_error)

    # Set compute_accuracy=False when using MSE.
    model.compute_accuracy = False

    # Define optimizer (Adam, RMSProp, etc)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(train, batch_size)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch',
        'main/loss',
    ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

    # Run trainer
    trainer.run()

    # Should print out value close to W.
    print(model.predictor(np.ones((1, n_in)).astype(np.float32)).data)

if __name__ == "__main__":
    main()

