import torch

def flops_lat_fn(conv, input_shape):
    conv_filter = conv.weight.shape
    stride = conv.stride[0]
    padding = conv.padding[0]
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    flops_per_instance = n + (n-1)    # general defination for number of flops (n: multiplications and n-1: additions)
    num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2*padding ) / stride ) + 1  # for rows
    num_instances_per_filter *= (( input_shape[1] - conv_filter[2] + 2*padding ) / stride ) + 1 # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]    # multiply with number of filters
    return total_flops_per_layer
