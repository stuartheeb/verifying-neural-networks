import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

def get_flattened_shape(input_shape):
    flattened_input_shape = 1
    for dim in range(len(input_shape)):
        flattened_input_shape *= input_shape[dim]
    return flattened_input_shape

class DeepPolyConstraints:
    def __init__(self, lbounds, ubounds):
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.previous = None
        self.transformer = None
    
    @classmethod
    def constraints_from_eps(cls, inputs, eps, clipper):
        return cls((inputs-eps).clamp(*clipper), (inputs+eps).clamp(*clipper))
    
    @classmethod
    def constraints_from_transformer(cls, previous, lslope, lintercept, uslope, uintercept):
        self = cls.__new__(cls)
        self.lbounds = None
        self.ubounds = None
        self.previous = previous
        self.transformer = (lslope, lintercept, uslope, uintercept)
        return self

    @classmethod
    def constraints_from_flatten(cls, previous, lbounds, ubounds):
        self = cls.__new__(cls)
        self.lbounds = lbounds
        self.ubounds = ubounds
        self.previous = previous
        self.transformer = "flatten"
        return self

    def certifier(self, lslope, lintercept, uslope, uintercept, initial_bounds):
        lslope_pos = torch.where(lslope > 0, lslope, torch.zeros_like(lslope))
        lslope_neg = torch.where(lslope < 0, lslope, torch.zeros_like(lslope))
        uslope_pos = torch.where(uslope > 0, uslope, torch.zeros_like(uslope))
        uslope_neg = torch.where(uslope < 0, uslope, torch.zeros_like(uslope))

        self.lbounds = initial_bounds.lbounds @ lslope_pos.T + initial_bounds.ubounds @ lslope_neg.T + lintercept
        self.ubounds = initial_bounds.ubounds @ uslope_pos.T + initial_bounds.lbounds @ uslope_neg.T + uintercept

    def backsubstitution(self):
        if self.lbounds is not None and self.ubounds is not None:
            return
        
        current = self
        clslope, clintercept, cuslope, cuintercept = current.transformer
        current = current.previous

        while current.transformer is not None:
            if current.transformer == "flatten":
                current = current.previous 
                continue
            
            plslope, plintercept, puslope, puintercept = current.transformer

            clslope_pos = torch.where(clslope > 0, clslope, torch.zeros_like(clslope))
            clslope_neg = torch.where(clslope < 0, clslope, torch.zeros_like(clslope))
            cuslope_pos = torch.where(cuslope > 0, cuslope, torch.zeros_like(cuslope))
            cuslope_neg = torch.where(cuslope < 0, cuslope, torch.zeros_like(cuslope))

            clintercept = plintercept @ clslope_pos.T + puintercept @ clslope_neg.T + clintercept
            cuintercept = puintercept @ cuslope_pos.T + plintercept @ cuslope_neg.T + cuintercept
            clslope = clslope_pos @ plslope + clslope_neg @ puslope
            cuslope = cuslope_pos @ puslope + cuslope_neg @ plslope
            
            current = current.previous

        self.certifier(clslope, clintercept, cuslope, cuintercept, current)
        
class DeepPolyLinearLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.W = layer.weight.detach()
        self.b = layer.bias.detach()

    def forward(self, previous):
        return DeepPolyConstraints.constraints_from_transformer(previous, self.W, self.b, self.W, self.b)

class DeepPolyFlattenLayer(nn.Module):
    def __init__(self, layer, is_first = False):
        super().__init__()
        self.flatten = layer
        self.is_first = is_first

    def forward(self, previous):
        previous.backsubstitution()
        lbounds = previous.lbounds.flatten()
        ubounds = previous.ubounds.flatten()

        if self.is_first:
            return DeepPolyConstraints(lbounds, ubounds)
            
        return DeepPolyConstraints.constraints_from_flatten(previous, lbounds, ubounds)

class DeepPolyConv2D(nn.Module):
    def __init__(self, layer, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.W = layer.weight.detach()
        self.b = layer.bias.detach()
        self.stride = layer.stride
        self.padding = layer.padding

    def forward(self, previous):
        kernel, bias = self.W, self.b
        stride, padding = self.stride, self.padding
        input_shape = self.input_shape
        flattened_input_shape = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

        W = torch.eye(flattened_input_shape).view(list(input_shape) + [flattened_input_shape]).permute(0, 1, 4, 2, 3)
        W = nn.functional.conv3d(W, kernel.unsqueeze(2), stride=tuple([1] + list(stride)), padding=tuple([0] + list(padding))).permute(0, 1, 3, 4, 2)
        b = torch.ones(W.shape[:-1])
        b = b * bias[:, None, None]

        W = torch.flatten(W[0], start_dim=0, end_dim=2)
        b = torch.flatten(b)

        return DeepPolyConstraints.constraints_from_transformer(previous, W, b, W, b)

class DeepPolyReLU(nn.Module):
    def __init__(self, layer, input_shape):
        super().__init__()
        self.flattened_input_shape = get_flattened_shape(input_shape)
        self.raw_alpha = nn.Parameter(data=torch.ones(self.flattened_input_shape), requires_grad=True)

    def forward(self, previous):
        alpha = torch.sigmoid(self.raw_alpha)

        previous.backsubstitution()
        lbounds = previous.lbounds
        ubounds = previous.ubounds
        N = self.flattened_input_shape

        relu_slope = (ubounds / (ubounds - lbounds)).flatten()
        relu_slope[relu_slope != relu_slope] = 0
        relu_intercept = (1 - relu_slope) * ubounds

        # Case 1: ubounds <= 0
        below_nodes = (ubounds <= 0).flatten()
        lslope, lintercept = torch.zeros(N), torch.zeros(1, N)
        uslope, uintercept = torch.zeros(N), torch.zeros(1, N)

        # Case 2: lbounds >= 0
        above_nodes = (lbounds >= 0).flatten()
        lslope = torch.where(above_nodes, torch.ones(N), lslope)
        uslope = torch.where(above_nodes, torch.ones(N), uslope)

        # Case 3: crossing ReLUs
        crossing_nodes = ~(below_nodes | above_nodes)

        uslope = torch.where(crossing_nodes, relu_slope, uslope)
        uintercept = torch.where(crossing_nodes, relu_intercept, uintercept)
        # V1
        lslope_v1 = torch.where(crossing_nodes, torch.zeros(N), lslope)
        # V2
        lslope_v2 = torch.where(crossing_nodes, torch.ones(N), lslope)

        lslope = alpha*lslope_v1 + (1-alpha)*lslope_v2

        lslope = torch.diag(lslope)
        uslope = torch.diag(uslope)

        return DeepPolyConstraints.constraints_from_transformer(previous, lslope, lintercept, uslope, uintercept)
    
class DeepPolyLeakyReLU(nn.Module):
    def __init__(self, layer, input_shape):
        super().__init__()
        self.negative_slope = layer.negative_slope
        self.flattened_input_shape = get_flattened_shape(input_shape)
        self.raw_alpha = nn.Parameter(data=torch.ones(self.flattened_input_shape), requires_grad=True)

    def forward(self, previous):
        alpha = torch.sigmoid(self.raw_alpha)
        neg_slope = self.negative_slope

        previous.backsubstitution()
        lbounds = previous.lbounds
        ubounds = previous.ubounds
        N = self.flattened_input_shape

        leaky_relu_slope = ((ubounds - lbounds*neg_slope) / (ubounds - lbounds)).flatten()
        leaky_relu_slope[leaky_relu_slope != leaky_relu_slope] = 0
        leaky_relu_intercept = (1 - leaky_relu_slope) * ubounds

        # Initialization
        lslope, lintercept = torch.ones(N), torch.zeros(1, N)
        uslope, uintercept = torch.ones(N), torch.zeros(1, N)

        # Case 1: ubounds <= 0
        below_nodes = (ubounds <= 0).flatten()
        lslope = torch.where(below_nodes, neg_slope * torch.ones(N), lslope)
        uslope = torch.where(below_nodes, neg_slope * torch.ones(N), uslope)

        # Case 2: lbounds >= 0
        above_nodes = (lbounds >= 0).flatten()
        lslope = torch.where(above_nodes, torch.ones(N), lslope)
        uslope = torch.where(above_nodes, torch.ones(N), uslope)

        # Case 3: crossing leaky ReLUs (triangular shape, differentiate if neg_slope > 1)
        crossing_nodes = ~(below_nodes | above_nodes)

        if neg_slope <= 1:
            uslope = torch.where(crossing_nodes, leaky_relu_slope, uslope)
            uintercept = torch.where(crossing_nodes, leaky_relu_intercept, uintercept)
            
            # V1
            lslope_v1 = torch.where(crossing_nodes, torch.ones(N), lslope)
            # V2
            lslope_v2 = torch.where(crossing_nodes, neg_slope*torch.ones(N), lslope)

            lslope = alpha*lslope_v1 + (1-alpha)*lslope_v2

        elif neg_slope > 1:
            lslope = torch.where(crossing_nodes, leaky_relu_slope, lslope)
            lintercept = torch.where(crossing_nodes, leaky_relu_intercept, lintercept)

            # V1
            uslope_v1 = torch.where(crossing_nodes, torch.ones(N), uslope)
            # V2
            uslope_v2 = torch.where(crossing_nodes, neg_slope*torch.ones(N), uslope)

            uslope = alpha*uslope_v1 + (1-alpha)*uslope_v2

        lslope = torch.diag(lslope)
        uslope = torch.diag(uslope)

        return DeepPolyConstraints.constraints_from_transformer(previous, lslope, lintercept, uslope, uintercept)

class DeepPolySequential(nn.Sequential):
    def __init__(self, network, inputs):
        def get_dp_layer(layer, input_shape):
            if isinstance(layer, nn.Linear):
                return DeepPolyLinearLayer(layer)
            elif isinstance(layer, nn.Conv2d):
                return DeepPolyConv2D(layer, input_shape)
            elif isinstance(layer, nn.ReLU):
                return DeepPolyReLU(layer, input_shape)
            elif isinstance(layer, nn.LeakyReLU):
                return DeepPolyLeakyReLU(layer, input_shape)
            elif isinstance(layer, nn.Flatten):
                return DeepPolyFlattenLayer(layer)
            elif isinstance(layer, nn.Sequential): # check if this is needed
                return DeepPolySequential(layer, inputs)
            else:
                raise NotImplementedError(f"No DP layer for layer {layer}")

        current = inputs
        layers = [DeepPolyFlattenLayer(nn.Flatten, is_first = True)]
        for child in network.children():
            layers.append(get_dp_layer(child, current.shape))
            current = child(current)

        super().__init__(*layers)


class DeepPolyVerifier(nn.Module):
    def __init__(self, true_label, num_labels = 10):
        super().__init__()
        W = -torch.eye(num_labels)
        W = W[torch.arange(num_labels) != true_label, :]
        W[:, true_label] = torch.ones(num_labels - 1)
        self.W = W
        self.b = torch.zeros(1, num_labels - 1)

    def forward(self, previous):
        out_shape = DeepPolyConstraints.constraints_from_transformer(previous, self.W, self.b, self.W, self.b)
        out_shape.backsubstitution()
        return out_shape

class DeepPolyLoss(nn.Module):
    def forward(self, previous):
        lbounds = previous.lbounds
        return torch.log(-lbounds[lbounds < 0]).max()
