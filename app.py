# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import io
import sys
from torch import nn
import torch.nn.functional as F
from flask import Flask,request,jsonify,render_template
app = Flask(__name__, static_url_path='/static')

class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        ## pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = False
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
        


@app.route('/',methods=['GET'])
def g():
        return render_template('index.html', sentiment='')

@app.route('/about',methods=['GET'])
def about():
        return render_template('about.html', sentiment='')

@app.route('/predict',methods=['POST','GET'])
def generate():
    if request.method == 'POST':
        data = request.form.get('text')
        prime = data
        top_k = 5
        size = 100
        for ch in prime : 
            if not ('\u0600' <= ch <= '\u06FF' ):
                a = {"error":"جميع الحروف يجب ان تكون عربية",
                     "error2":"All letters should be in Arabic"}
                return render_template('index.html', sentiment=a ,title=data)

        try :
            size = int(request.form.get('size'))
        except Exception:
            a = {"error":"عدد الحروف يجب ان يكون عددأ صحيحأ",
                     "error2":"size should be a valid number"}
            return render_template('index.html', sentiment=a ,title=data)

        print(net.load_state_dict(checkpoint['state_dict']))

        net.cpu()
        net.eval() # eval mode
        train_on_gpu  =False
        # First off, run through the prime characters
        chars = [ch for ch in prime]
        h = net.init_hidden(1)
        for char in prime:
            # tensor inputs
            x = np.array([[net.char2int[char]]])
            n_labels = len(net.chars)
                # Initialize the the encoded array
            one_hot = np.zeros((x.size, n_labels), dtype=np.float32)

            # Fill the appropriate elements with ones
            one_hot[np.arange(one_hot.shape[0]), x.flatten()] = 1.

            # Finally reshape it to get back to the original array
            one_hot = one_hot.reshape((*x.shape, n_labels))

            
            
            x = one_hot
            inputs = torch.from_numpy(x)
            
            if(train_on_gpu):
                inputs = inputs.cuda()
            
            # detach hidden state from history
            h = tuple([each.data for each in h])
            # get the output of the model
            out, h = net(inputs, h)

            # get the character probabilities
            p = F.softmax(out, dim=1).data
            if(train_on_gpu):
                p = p.cpu() # move to cpu
            
            # get top characters
            if top_k is None:
                top_ch = np.arange(len(net.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()
            
            # select the likely next character with some element of randomness
            p = p.numpy().squeeze()
            char = np.random.choice(top_ch, p=p/p.sum())
            
            char , h = net.int2char[char], h
            #char, h = predict(net, ch, h, top_k=top_k)

        chars.append(char)
        
        # Now pass in the previous character and get a new one
        for ii in range(size):
                x = np.array([[net.char2int[char[-1]]]])
                n_labels = len(net.chars)
                        # Initialize the the encoded array
                one_hot = np.zeros((x.size, n_labels), dtype=np.float32)

                # Fill the appropriate elements with ones
                one_hot[np.arange(one_hot.shape[0]), x.flatten()] = 1.

                # Finally reshape it to get back to the original array
                one_hot = one_hot.reshape((*x.shape, n_labels))

                x = one_hot
                inputs = torch.from_numpy(x)
                
                if(train_on_gpu):
                    inputs = inputs.cuda()
                
                # detach hidden state from history
                h = tuple([each.data for each in h])
                # get the output of the model
                out, h = net(inputs, h)

                # get the character probabilities
                p = F.softmax(out, dim=1).data
                if(train_on_gpu):
                    p = p.cpu() # move to cpu
                
                # get top characters
                if top_k is None:
                    top_ch = np.arange(len(net.chars))
                else:
                    p, top_ch = p.topk(top_k)
                    top_ch = top_ch.numpy().squeeze()
                
                # select the likely next character with some element of randomness
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p/p.sum())
                
                # return the encoded value of the predicted char and the hidden state
                #return net.int2char[char], h
                
                char, h = net.int2char[char], h
                print(char)
                chars.append(char)
        output = ''.join(chars)    
        reshaped_text = output     
  

        a = {}
        s =""
        for ii in range(len(reshaped_text.split(' '))):
            s += reshaped_text.split(' ')[ii]+" "
            if ii %5 ==0 :
                a[ii] = s
                s = ""

        #return reshaped_text
        return render_template('index.html', sentiment=a ,title=data)
    else:    
        return render_template('index.html', sentiment='')

if __name__ == '__main__':
    global net
    with open('rnn_50_epoch.net', 'rb') as f:
        checkpoint = torch.load(f,map_location=torch.device('cpu'))
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    port = int(os.environ.get("PORT", 5000))

    app.run( debug=True, host='0.0.0.0', port=port)
