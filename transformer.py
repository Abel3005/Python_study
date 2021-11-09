import torch 
import numpy as np
import matplotlib.pyplot as plt
class PositionalEncoding:
    def __init__(self):
        return
    # position: num_row * 1 i = 1 * d_model
    def getAngle(self, d_model):
        angle = 1 / torch.pow(10000, (2*(torch.arange(d_model,dtype=torch.float32).unsqueeze(0)//2)) / d_model)
        return angle
    def positional_encoding(self, num_row, d_model):
        position = torch.arange(num_row,dtype=torch.float32).unsqueeze(1)
        result = position.matmul(self.getAngle(d_model))
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:,1::2])
        return result
    
class Transformer:
    def __init__(self):
        self.d_model = 512
        self.num_layer = 6
        self.num_heads = 8
        self.num_roows = 50
        self.d_ff = 2048
        self.posional_encoder = PositionalEncoding()
        return
    def showPositionEmbadding(self):
        x = self.posional_encoder.positional_encoding(self.num_roows,self.d_model)
        plt.pcolormesh(x.numpy(),cmap='RdBu')
        plt.xlim((0,self.d_model))
        plt.xlabel('depth')
        plt.ylabel('position')
        plt.colorbar()
        plt.show()
        return
def main():
    model = Transformer()
    model.showPositionEmbadding()
    return

if __name__ == "__main__":
    main()