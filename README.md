# Transformers
Transformers is the state-of-the-art neural network on NLP. First introduce by Google's paper "Attention is all you need" in 2017. It can perform much better than RNN in many application such as machine translation. The code was implemented base on pytorch class Transformer. It's easy to understand the structure of the transformer and you can build your own transformer or apply it directly.

<img src="https://github.com/chanyikchong/Transformers/blob/master/transformer.png" width="300"><br/>

## Package
You will need the following package to run the model
- numpy
- pytorch

## Usage
You can download the zip file or clone the repository to your computer.

The first step is to import the model
```python
import transformers as t
```
You can directly use a built transformer 
```python
transformer = t.Transformer(dim_model = 512, num_heads = 8, num_encoder_layer = 6, 
                  num_decoder_layer = 6, dim_ffc = 2048, dropout = 0.1,
                 activation = 'relu', normalize_before = False)
                 #these are the default values
output = transformer(input, target, pos = None, build_in_pos = True, query_pos = None)
#pos is the position tensor for the input, you can use a build in position tensor by setting pos = None and build_in_pos = True
#query_pos is the position tensor for the target
```
The model provide a build position encoder which use the sine and cosine function just like the paper did.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

