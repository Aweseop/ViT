# Vision transformer
This is a Pytorch reimplementation of Vision Transformer model in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby).

It aims to implement ViT simply using the MNIST dataset.
More than 95% performance with 10 epochs.


---

## Model structure
![image](![vit1](https://user-images.githubusercontent.com/87693860/177263779-d5efc6ea-7f32-424f-b645-a555a23827ab.PNG))

## Usage
```
# Train, test ViT Using the MNIST dataset
python main.py --in_channel 1 --img_size 28 --patch_size 4 --emb_dim 4*4  \
               --n_layers 6 --num_heads 2 --forward_dim 4 --dropout_ratio 0.1 --n_classes 10

# It can be executed simply as follows
bash train.sh

```

## Citations

```
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
