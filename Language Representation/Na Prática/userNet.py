import torch
import torch.nn as nn

class UserCommentsNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim):
        super(UserCommentsNet, self).__init__()
        
        # Camada de embedding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Camada convolucional
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        # Camada de max pooling
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        
        # Camada totalmente conectada
        self.fc = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        
        # Camada de saída
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text: (batch_size, seq_len)
        
        # Passar o texto pela camada de embedding
        embedded = self.embedding(text) # (batch_size, seq_len, embedding_dim)
        
        # Transpor para ter o shape (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Aplicar as camadas convolucionais
        conv_outs = [nn.functional.relu(conv(embedded)) for conv in self.conv_layers]
        
        # Aplicar max pooling
        pooled = [self.pooling(conv_out).squeeze(-1) for conv_out in conv_outs]
        
        # Concatenar as saídas das camadas convolucionais em uma única matriz
        cat = torch.cat(pooled, dim=1)
        
        # Passar a matriz concatenada pela camada totalmente conectada
        fc_out = nn.functional.relu(self.fc(cat))
        
        # Passar a saída da camada totalmente conectada pela camada de saída
        output = self.output(fc_out)
        
        return output


'''
    Rede com camada de embeddings, convolucional, max pooling e totalmente conectada 
    que recebe os comentários de um usuário e extrai um vetor de features unidimensional 
    de 128 posições.
    
    - num_embeddings é o número de palavras distintas no vocabulário;
    - embedding_dim é o tamanho dos vetores de embedding para cada palavra;
    - num_filters é o número de filtros a serem aplicados na camada convolucional;
    - filter_sizes é uma lista de tamanhos de filtros a serem usados na camada convolucional;
    - hidden_dim é o tamanho da camada oculta da camada totalmente conectada;
    - output_dim é o tamanho do vetor de features de saída.

'''