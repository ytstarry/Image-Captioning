import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # Initializing hidden state size (Number of nodes in the hidden layer)
        self.hidden_size = hidden_size # Hyperparameter - hidden size
        
        # Initializing vocab_size (The size of vocabulary or output size)
        self.vocab_size = vocab_size
        
        # Initializing embedding
        self.word_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        
        # Initializing LSTM: the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        # Initializing the output
        self.linear = nn.Linear(in_features = hidden_size, 
                                out_features = vocab_size) # Linear(in_features=hidden_size, out_features=8855, bias=True)
    
    def forward(self, features, captions):
        
        embeddings = self.word_embeddings(captions)
        
        # features contains the embedded image features; features.shape: torch.Size([10, 512])
        # embedding; captions.shape: torch.Size([10, 16, 512]) if caption length = 16
        # Concatenating features to embedding
        features = features.unsqueeze(1) # change shape from torch.Size([10, 512]) to torch.Size([10, 1, 512])
        embedding = torch.cat((features, embeddings[:, :-1,:]), dim=1) # embedding's size -> torch.Size([10, 17, 512]) 
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, hidden = self.lstm(embedding) # lstm_out's shape:  torch.Size([10, 17, 512])
        
        # get the scores from linear layer
        # 3.1 - Running through the linear layer
        outputs = self.linear(lstm_out) # output's shape: torch.Size([10, 17, 8855])
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # input already unsqueezed; input feature: (batch_size, 1, embed_size)
        
        predicted_sentence = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states) # lstm_output: (batch_size, 1, hidden_size)
            
            outputs = self.linear(lstm_out.squeeze(1)) # output: (batch_size, vocab_size)
            
            _, predicted = outputs.max(1) # select the vocab with max prob; predicted: (batch_size)
            
            predicted_sentence.append(predicted.item())
            
            inputs = self.word_embeddings(predicted) # inputs: (batch_size, embed_size)
            
            inputs = inputs.unsqueeze(1) # inputs: (batch_size, 1, embed_size)
       
        #predicted_sentence = torch.stack(predicted_sentence, 1) 
        
        return predicted_sentence # predicted_sentence: (batch_size, max_seq_length)
        """
        sampled_ids = []
        #inputs = inputs.unsqueeze(1)
        for i in range(max_len):
            #LSTM cell h, c
            hidden, states = self.lstm(inputs,states)
            outputs = self.linear(hidden.squeeze(1)) 
            #arg max probability per output in LSTM cell 
            _, predicted = outputs.max(1)    
            sampled_ids.append(predicted)
            #Update Hidden state with new output to next LSTM cell
            #How to tell if the index is word-vector index?
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1) 
            
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = list(sampled_ids.cpu().numpy()[0])
        sampled_ids = [int(i) for i in sampled_ids]
        return  sampled_ids
        """