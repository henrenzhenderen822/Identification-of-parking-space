# -Environment:
python3.9  pytorch1.10.2   cuda10.2   opencv4.5.5  
# -Folder:
cameras: The image of parking lot and the corresponding coordinates of parking space are established.    
checkpoints: The folder is used to store the trained model.  
datasets: The folder are used to store data sets and corresponding labels (txt format).  
net: The folder the network structure for various resolution inputs.    
result: The folder is used to save the results of the test images.  
datasets: The file in the folder is used to build the dataset.  
# -File:
mark.py: File to mark and return the coordinates of the parking space (images and return results are saved in the cameras folder).    
run.py: Files are used for testing (the results are saved in the result folder).  
parameters.py: Used to configure parameters.  
train.py: For training networks.  
auto_script.py: Used to automatically modify parameter running training.  
# -Data：  
MiniPK：见MiniPK文件夹  
PKLot：http://web.inf.ufpr.br/vri/parking-lot-database  
CNRPark：http://claudiotest.isti.cnr.it/park-datasets/
