import tensorflow as tf
import os
from model import train, test, output,write_output_in_files
from configuration import get_config

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":
    # start training
    if config.train:
        print("\nTraining Session")
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        train(config.model_path)
    # start test
    else:
        if config.test:
            print("\nTest session")
            test(config.model_path)
        #if os.path.isdir(config.model_path):
        else:
            #output(config.model_path)
            write_output_in_files(config.model_path):
        #else:
        #    raise AssertionError("model path doesn't exist!")
