
```
# https://stackoverflow.com/questions/73911967/how-to-use-pytorch-multi-head-attention-for-classification-task?rq=3
attention_layer = nn.MultiHeadAttion(300 , 300%num_of_heads==0,dropout=0.1)
neural_net_output = point_wise_neural_network(attention_layer)
normalize = LayerNormalization(input + neural_net_output)
globale_average_pooling = nn.GlobalAveragePooling(normalize)
nn.Linear(input , num_of_classes)(global_average_pooling)
```