"""
Author: jose.saavedra
This is an example for training on a sketch classification problem
model_dir and data_dir should changed to the correct paths
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.mlp as mlp

if __name__ == "__main__" :
    params = { "device" : "/gpu:0",
              "model_dir" : "/home/vision/smb-datasets/SBIR/QuickDraw-Animals/models",
              "data_dir" : "/home/vision/smb-datasets/SBIR/QuickDraw-Animals",              
              "learning_rate" : 0.001,  
              "number_of_classes" : 12,
              "number_of_iterations" : 40000,
              "batch_size" : 80,
              "data_size" : 12000,
        }               
    my_mlp = mlp.MLP(params)
    print("MLP initialized ok")
    print("--------start training")
    my_mlp.train()
    #Use my_mlp.test() for testing
    #Use my_mlp.save_model() for saving models that will be used for fast_prediction
    print("--------end training")
    