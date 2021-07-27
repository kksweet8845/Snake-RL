from .models.dqn import DQN_DNN






if __name__ == "__main__":



    dqn = DQN_DNN(24, 5)

    for param_tensor in dqn.state_dict():
        print(param_tensor, "\t", dqn.state_dict()[param_tensor].size())