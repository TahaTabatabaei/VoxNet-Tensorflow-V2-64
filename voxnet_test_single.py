import math, datetime
from voxnet import *
from volumetric_data import ShapeNet40Vox30
# import tensorflow as tf

# Load the dataset and initialize the model
dataset = ShapeNet40Vox30()
voxnet = VoxNet()

# Placeholders
p = dict()
p['labels'] = tf.placeholder(tf.float32, [None, 40])
p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))



# Load the checkpoint number
checkpoint_num = 30  # Use the specified checkpoint

# Function to test a single sample
def test_single_sample():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session, 'checkpoints/c-{}.npz'.format(checkpoint_num))
        
        # Load a random single sample
        # voxs, labels = dataset.test.get_batch(1)  # Get a batch with a single sample

        # or load a sample by file name
        voxs, labels = dataset.test.get_sample_by_filename("car_000000073_4.npy")

        feed_dict = {voxnet[0]: voxs, p['labels']: labels}
        
        # Calculate accuracy
        s = session.run(p, feed_dict=feed_dict)
        print('Test accuracy for the single sample: {}'.format(s['accuracy']))

        # reall label vs predicted label
        print(f'The actual lable is: {np.argmax(labels[0])+1}')
        print(f'The predicted label is: {np.argmax(s["labels"][0])+1}')



# Run the test
test_single_sample()
