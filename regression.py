from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import tempfile
import subprocess

import automobile_data

log_dir = tempfile.mkdtemp()
print("tensorbord-dir", log_dir)

subprocess.Popen(['pkill', '-f', 'tensorboard'])
subprocess.Popen(['tensorboard', '--logdir', log_dir])

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000., type=float, help='price normalization factor')


def main(argv):
    """Builds, trains, and evaluates the model."""
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = automobile_data.load_data()

    train_y /= args.price_norm_factor
    test_y /= args.price_norm_factor

    # Build the training dataset.

    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, batch_size=64,
                                                            shuffle=True, num_epochs=None)

    # Build the validation dataset.

    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_x, y=test_y, batch_size=64, shuffle=False)

    # Build the Estimator.
    # model = tf.estimator.LinearRegressor(feature_columns=automobile_data.features_columns(), model_dir=log_dir)

    model = tf.estimator.DNNRegressor(hidden_units=[50,30,10], feature_columns=automobile_data.features_columns(),
                                      model_dir=log_dir)

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    model.train(input_fn=training_input_fn, steps=args.train_steps)

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=eval_input_fn)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}".format(args.price_norm_factor * average_loss ** 0.5))

    # Run the model in prediction mode.

    df = test_x[:2]
    pre_input_fn = tf.estimator.inputs.pandas_input_fn(x=df, shuffle=False)
    predict_results = model.predict(input_fn=pre_input_fn)

    # Print the prediction results.
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        print(args.price_norm_factor * prediction['predictions'])

    print()


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
