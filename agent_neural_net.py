import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Provide tile and entity observations to receive neural net input
def get_input(agent, tile, entity, pos):


  state = np.array([agent.health.val, agent.food.val, agent.water.val])

  tile = tile[:,2]    ##(15,15)
  entity = entity[:,[0,1,2,3,4,8,9,12,13,14]]
  entity[1] -= pos[0]
  entity[2] -= pos[1] #(25,10)
  return tile.reshape(1,15,15), entity.reshape(1,25,10), state.reshape(1,1,3)


class PolicyNet(nn.Module):
    def __init__(self, output_size_move, output_size_attack, hidden_size=16):
        super(PolicyNet, self).__init__()

        # Define the CNN for the tile input
        self.tile_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Assuming the output of the conv layers is flattened to size 32*1*1=32 for simplicity
        self.tile_conv_out_size = 72

        # Define the fully connected layers for the entity input
        self.entity_fc = nn.Sequential(
            nn.Linear(in_features=25*10+3, out_features=32),
            nn.Tanh()
        )

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.tile_conv_out_size + 32, hidden_size=hidden_size, batch_first=True)
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))

        # Define the final linear layers
        self.fc_move = nn.Linear(in_features=hidden_size, out_features=output_size_move)
        self.fc_style = nn.Linear(in_features=hidden_size, out_features=3)
        self.fc_attack = nn.Linear(in_features=hidden_size, out_features=output_size_attack)

        self.output_size_move = output_size_move
        self.output_size_attack = output_size_attack

    def forward(self, inp):
        with torch.no_grad():
          # Process the tile input
          tile, entity, state = inp

          tile = torch.from_numpy(tile).float()
          entity = torch.from_numpy(entity).float()
          state = torch.from_numpy(state).float()

          tile = tile.unsqueeze(1)  # Add channel dimension
          tile_out = self.tile_conv(tile)

          # Process the entity input
          entity = entity.reshape(entity.size(0), -1)  # Flatten entity input
          state = state.reshape(state.size(0), -1)  # Flatten entity input
          entity = torch.cat((entity, state), dim=-1)
          entity_out = self.entity_fc(entity)

          # Concatenate tile and entity outputs
          combined = torch.cat((tile_out, entity_out), dim=1).unsqueeze(1)  # Add sequence dimension

          # Pass through LSTM layer
          lstm_out, hidden = self.lstm(combined, self.hidden)
          self.hidden = hidden
          lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

          # Compute the move and attack logits
          move_logits = self.fc_move(lstm_out)
          attack_logits = self.fc_attack(lstm_out)
          style = torch.argmax(self.fc_style(lstm_out)).item()
          move_action = torch.argmax(move_logits.flatten())
          attack_action = torch.argmax(attack_logits.flatten())

          # Convert logits to categorical probability distributions
          #move_probs = F.softmax(move_logits/0.5, dim=1)
          #attack_probs = F.softmax(attack_logits/0.5, dim=1)

          # Sample from the distributions
          #move_action = torch.multinomial(move_probs, num_samples=1)
          #attack_action = torch.multinomial(attack_probs, num_samples=1)

        return move_action.squeeze(-1), style, attack_action.squeeze(-1)


class NoCombatNet(nn.Module):
    def __init__(self, output_size_move, output_size_attack, hidden_size=64):
        super(NoCombatNet, self).__init__()

        # Define the CNN for the tile input
        self.tile_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Assuming the output of the conv layers is flattened to size 32*1*1=32 for simplicity
        self.tile_conv_out_size = 288


        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.tile_conv_out_size, hidden_size=hidden_size, batch_first=True)
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))

        # Define the final linear layers
        self.fc_move = nn.Linear(in_features=hidden_size, out_features=output_size_move)

        self.output_size_move = output_size_move

    def forward(self, inp):
        with torch.no_grad():
          # Process the tile input
          tile, entity = inp

          tile = torch.from_numpy(tile).float()
          entity = torch.from_numpy(entity).float()

          tile = tile.unsqueeze(1)  # Add channel dimension
          tile_out = self.tile_conv(tile).unsqueeze(1)

          # Pass through LSTM layer
          lstm_out, hidden = self.lstm(tile_out, self.hidden)
          self.hidden = hidden
          lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

          # Compute the move and attack logits
          move_logits = self.fc_move(lstm_out)

          move_probs = F.softmax(move_logits/0.5, dim=1)

          # Sample from the distributions
          move_action = torch.multinomial(move_probs, num_samples=1)

        return move_action.squeeze(-1), None




# Functions for saving and loading neural network weights
def save_state(models_dictionary, model_dict, save_path):
  for i in models_dictionary:
    torch.save(model_dict[i].state_dict(), f"{save_path}/agent_{i}")
    #torch.save(model_dict[i][1].state_dict(), f"{save_path}/agent_attack_{i}")

def load_state(models_dictionary, load_path):
  for i in models_dictionary:
    model_dict[i][0].load_state_dict(torch.load(f"{load_path}/agent_{i}"))
    #model_dict[i][1].load_state_dict(torch.load(f"{load_path}/agent_attack_{i}"))

