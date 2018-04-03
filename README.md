# DCGAN-Tensorflow
SImple Tensorflow implementation of "Deep Convolutional Generative Adversarial Networks" 

## Usage
### Download Dataset
```bash
> python download.py celebA lsun
```

* `mnist, cifar10` are in keras... So, you don't have to download it


```bash
> python main.py --dataset celebA
```

## Results (epoch = 20, lr = 0.0002, batch_size = 64, z_dim = 100)
### mnist
![mnist](./assests/mnist.png)
![mnist_loss](./assests/mnist_loss.png)

### celebA
![celebA](./assests/celebA.png)
![celebA_loss](./assests/celebA_loss.png)

### cifar10
![cifar10](./assests/cifar10.png)
![cifar10_loss](./assests/cifar10_loss.png)

## Author
Junho Kim
