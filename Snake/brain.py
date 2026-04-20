import torch
import torch.nn as nn
# import torch.nn.functional as F
    
class actor_mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, seed):
        super(actor_mlp, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared feature learning layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.feature_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        
        self.output_layer = nn.Linear(hidden_size[-1], output_size)

        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Pass through shared feature layers
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        x = self.output_layer(x)
        
        # use softmax activation to get probabilities
        x = self.output_activation(x)
        
        return x
    

class actor_cnn(nn.Module):
    def __init__(self, input_size, output_size, cnn_filters, kernel_sizes, mlp_head_size, seed, state_shape):
        super(actor_cnn, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_shape = state_shape

        self.conv_layers = nn.ModuleList()
        in_ch, current_h, current_w = self.state_shape

        # Convolutional layers
        for i in range(len(cnn_filters)):
            padding = 1 
            stride = 1
            self.conv_layers.append(
                nn.Conv2d(in_ch, cnn_filters[i], kernel_size=kernel_sizes[i], stride=stride, padding=padding)
            )
            
            # Update spatial dimensions (Height and Width)
            kernel_size = kernel_sizes[i]
            current_h = ((current_h + 2 * padding - kernel_size) // stride) + 1
            current_w = ((current_w + 2 * padding - kernel_size) // stride) + 1
            in_ch = cnn_filters[i]

        # Calculate the flattened size: Filters * Final_Height * Final_Width
        self.flattened_size = cnn_filters[-1] * current_h * current_w
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.flattened_size, mlp_head_size[0]))
        for i in range(len(mlp_head_size)-1):
            self.fc_layers.append(nn.Linear(mlp_head_size[i], mlp_head_size[i+1]))
        
        self.output_layer = nn.Linear(mlp_head_size[-1], output_size)
        
        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Reshape flattened input to (Batch, C, H, W)
        x = x.view(-1, *self.state_shape)

        for conv in self.conv_layers:
            x = self.activation(conv(x))
        
        # Flatten before the FC layers
        x = x.view(x.size(0), -1) 
        
        for fc in self.fc_layers:
            x = self.activation(fc(x))

        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    