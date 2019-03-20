import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.fast_predictor as fp

if __name__ == "__main__" :  
    params = { "model_dir" : "/home/vision/smb-datasets/SBIR/QuickDraw-Animals/models",
               "data_dir" : "/home/vision/smb-datasets/SBIR/QuickDraw-Animals",
               "device" : "/gpu:0",                
               "number_of_classes" : 12
          }
    
#     params = { "device" : "/gpu:0",
#               "model_dir" : "/home/vision/smb-datasets/MNIST-small/models",
#               "data_dir" : "/home/vision/smb-datasets/MNIST-small",                          
#               "number_of_classes" : 10              
#         }
    predictor = fp.FastPredictor(params)
    while True :
        filename = input("Image: ")    
        prediction = predictor.predict(filename)
        print(prediction)
        
        