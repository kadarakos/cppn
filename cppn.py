import torch
import torch.nn as nn
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
import argparse
import matplotlib
from pylab import savefig
import numpy as np
import imageio

parser = argparse.ArgumentParser(description='Create a Simple CPPN')

# Path Arguments
parser.add_argument('--random_seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--dim', type=int, default=500)
parser.add_argument('--cchannel', type=int, default=3)
parser.add_argument('--z_type', choices=['uniform', 'normal', 'none', 'sphere', 'gradual-normal'], default='normal')
parser.add_argument('--z_dim', type=int, default=32)
parser.add_argument('--scaling', type=int, default=8)
parser.add_argument('--layer_size', type=int, default=32)
parser.add_argument('--num_hidden', type=int, default=3)
parser.add_argument('--num_frames', type=int, default=1)

args = parser.parse_args()
print(args)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

def sample_z(method, frames, size):
    if method == 'uniform':
        z = np.random.uniform(-1, 1, size=(frames, 1, size)).astype(np.float32)
    elif method == 'normal':
        z = np.random.standard_normal(size=(frames, 1,  size)).astype(np.float32)
    elif method == 'none':
        z = np.zeros((frames, 1, size)).astype(np.float32)
    elif method == 'sphere':
        norm = np.random.normal
        normal_deviates = np.random.standard_normal(size=(frames, size))
        radius = np.sqrt((normal_deviates**2).sum(axis=0))
        points = normal_deviates/radius
        z = np.expand_dims(points, axis=1)
    elif method == "gradual-normal":
        total_frames = frames + 2
        z = np.zeros((total_frames, size))
        z1 =  np.random.standard_normal(size=size).astype(np.float32)
        z2 =  np.random.standard_normal(size=size).astype(np.float32)
        delta = (z2-z1) / (frames+1)
        for i in range(total_frames):
            if i == 0:
                z[i] = z1
            elif i == total_frames - 1:
                z[i] = z2
            else:
                z[i] = z1 + delta * float(i)
        z = np.expand_dims(z, axis=1)
    return z


class CustomActivationFunction(nn.Module):

    def __init__(self, mean=0, std=1, min=-0.9, max=0.9):
        super(CustomActivationFunction, self).__init__()
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.gauss = lambda x: torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))
        self.cos = torch.cos
        self.func = np.random.choice([self.gauss, self.cos])
    
    def forward(self, x):
        x = self.func(x)
        return torch.clamp(x, min=self.min, max=self.max)

    

class CPPN(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size=1, z_dim = 32, c_dim = 1, scale = 1.0, net_size = 32, num_hidden = 4):
        super(CPPN, self).__init__()
        self.batch_size = batch_size
        self.net_size = net_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_hidden = num_hidden
        #Build NN graph
        self.linear1 = nn.Linear(z_dim, self.net_size)
        self.linear2 = nn.Linear(1, self.net_size, bias=False)
        self.linear3 = nn.Linear(1, self.net_size, bias=False)
        self.linear4 = nn.Linear(1, self.net_size, bias=False)        
        self.linear8 = nn.Linear(self.net_size, self.c_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.custom = CustomActivationFunction()
        self.activations = [self.tanh, self.softsign, self.custom, self.relu]
        #self.lin_seq = nn.Sequential(self.tanh, self.linear5, self.tanh, self.linear6, self.tanh,
        #                         self.linear7, self.tanh, self.linear8, self.sigmoid)
        self.lin_seq = nn.Sequential()
        self.lin_list = [self.linear1, self.linear2, self.linear3, self.linear4]
        self.lin_list += [nn.Linear(self.net_size, self.net_size) for x in range(self.num_hidden)]
        self.lin_list += [self.linear8]
        #self.lin_list = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5, self.linear6, 
        #            self.linear7, self.linear8]
        self.generate_structure()
        for layer in self.lin_list:
            layer.weight.data.normal_(0, 1)
            try:
                layer.bias.data.fill_(0)
            except:
                pass
        print(self.lin_seq)

    def generate_structure(self):
        c = -1
        for l, layer in enumerate(self.lin_list[4:]):
            i = np.random.randint(0, len(self.activations))
            activation = self.activations[i]
            c += 1
            self.lin_seq.add_module(str(c), activation)
            c += 1
            self.lin_seq.add_module(str(c), layer)
        self.lin_seq.add_module(str(c+1), self.sigmoid)
        

    def forward(self, x, y, r, z):
        
        U = self.linear1(z) + self.linear2(x) + self.linear3(y) + self.linear4(r)   
        
        result = self.lin_seq(U)
        return result
    
def get_coordinates(x_dim = 32, y_dim = 32, scale = 10.0, batch_size = 1):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim
    
    # creates a list of x_dim values ranging from -1 to 1, then scales them by scale
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5 
    #x_range = x_range **2 
    #y_range = y_range **2
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    #r_mat = np.cos(x_mat) - np.cos(y_mat)
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()
         
def save_image(image_data, f, c_dim = 1):
    '''
    image_data is a tensor, in [height width depth]
    image_data is NOT the PIL.Image class
    '''
    plt.subplot(1, 1, 1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    if c_dim > 1:
      plt.imshow(image_data, interpolation='nearest')
    else:
      plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    ##plt.show()
    
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    fname = str(f) + str(args.cchannel) + '_' + str(args.random_seed) + '_' + str(args.layer_size) + '.png'
    savefig(fname, bbox_inches='tight', pad_inches=0.0)
    return fname

if __name__ == "__main__":
   
    model = CPPN(x_dim = args.dim, y_dim = args.dim, c_dim = args.cchannel, z_dim = args.z_dim, net_size = args.layer_size, num_hidden = args.num_hidden)
    x, y, r = get_coordinates(x_dim = args.dim, y_dim = args.dim, scale = args.scaling)   
    z = sample_z(args.z_type, args.num_frames, args.z_dim)
    with imageio.get_writer('movie.mp4', mode='I') as writer:
        for i in range(args.num_frames):
            print(i)
            z_scaled = V(torch.from_numpy(np.matmul(np.ones((args.dim*args.dim, 1)), z[i])).float())
            result = model.forward(x, y, r, z_scaled).squeeze(0).view(args.dim, args.dim, args.cchannel).data.numpy()
            fname = save_image(result, i, c_dim = args.cchannel)
            image = imageio.imread(fname)
            writer.append_data(image)
