import os, logging
import tarfile
import pickle
import argparse
import torch


from ml import RatioEstimator

logger = logging.getLogger(__name__)

#################################################
# Argument parsing
parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying the trained model - used in folder naming and input dataset naming')
parser.add_argument('-e', '--nentries',  action='store', type=int, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
parser.add_argument('--cuda_visible_devices', action='store', nargs="+", dest='cuda_visible_devices',type=int, default=None, help='Set GPUs visible to CUDA for NN training. If None, training uses one of the available ones (or CPU if no GPU exists)')

opts = parser.parse_args()

n = opts.nentries
global_name = opts.global_name
cuda_visible_devices = ",".join(str(device) for device in opts.cuda_visible_devices) if opts.cuda_visible_devices is not None else None

#################################################

#################################################
# Loading of data from numpy arrays created during training

if os.path.exists(f"data/{global_name}/data_out.tar.gz"):
    tar = tarfile.open(f"data/{global_name}/data_out.tar.gz")
    tar.extractall()
    tar.close()

# Check if already pre-processed numpy arrays exist
if os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy'):
    logger.info(" Loaded existing datasets ")
    x='data/'+global_name+'/X_train_'+str(n)+'.npy'
    y='data/'+global_name+'/y_train_'+str(n)+'.npy'
    w='data/'+global_name+'/w_train_'+str(n)+'.npy'
    x0='data/'+global_name+'/X0_train_'+str(n)+'.npy'
    w0='data/'+global_name+'/w0_train_'+str(n)+'.npy'
    x1='data/'+global_name+'/X1_train_'+str(n)+'.npy'
    w1='data/'+global_name+'/w1_train_'+str(n)+'.npy'
    f = open('data/'+global_name+'/metaData_'+str(n)+".pkl","rb")
    metaData = pickle.load(f)
    f.close()
else:
    logger.error("did not find data for model global_name +'_carl_'+str(n)")

estimator=RatioEstimator()
estimator.scaling_method = "minmax"

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
estimator.load(f'models/{global_name}/{global_name}_carl_{str(n)}',global_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
estimator.model.to(device) # need to send the model to GPU since that's where the input tensors are sent to in save function
estimator.save(f'models/{global_name}/{global_name}_carl_{str(n)}', x, metaData, export_model = True, noTar=True)
