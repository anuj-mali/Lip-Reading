import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x):
        # Add padding to the input
        x_padded = F.pad(x, (self.padding[2], self.padding[2], 
                             self.padding[1], self.padding[1], 
                             self.padding[0], self.padding[0]))
        
        # Calculate output dimensions
        D_out = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        H_out = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        W_out = (x.shape[4] + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1
        
        # Initialize output tensor
        output = torch.zeros((x.shape[0], self.out_channels, D_out, H_out, W_out), device=x.device)
        
        # Perform convolution
        for i in range(D_out):
            for j in range(H_out):
                for k in range(W_out):
                    # Compute the start and end indices for the depth, height, and width
                    start_d = i * self.stride[0]
                    end_d = start_d + self.kernel_size[0]
                    start_h = j * self.stride[1]
                    end_h = start_h + self.kernel_size[1]
                    start_w = k * self.stride[2]
                    end_w = start_w + self.kernel_size[2]
                    
                    # Extract the window to apply the kernel on
                    x_slice = x_padded[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
                    
                    # Apply the convolution for each output channel
                    for n in range(self.out_channels):
                        output[:, n, i, j, k] = torch.sum(x_slice * self.weight[n, :, :, :, :], dim=(1, 2, 3, 4))
        
        # Add bias if it exists
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output
