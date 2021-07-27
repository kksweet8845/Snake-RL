from dqn import DQN
import torch





if __name__ == "__main__":
    d = torch.rand(1, 3, 10, 10)
    m = DQN(10, 10,  4)
    o = m(d)
    
    onnx_path = "dqn.onnx"
    torch.onnx.export(m, d, onnx_path)