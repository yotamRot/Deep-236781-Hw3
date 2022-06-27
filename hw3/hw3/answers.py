r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=512,
        seq_len=16,
        h_dim=1024,
        n_layers=2,
        dropout=0.25,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=4,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "The sun went down"
    temperature = 0.4
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split into sequence because of the following reasons:
1. Since the corpus might be very large then including all data and RNN network information will not fit in memory. Therfore splitting into sequences will    make the RNN network smaller and require less data avialble in RAM so it will prevent this issue.

2. If we train on entire corpus it will cause our RNN network to be very deep. Therfore it might cause vanishing gradient issue that will make our model untrainable.


"""

part1_q2 = r"""
**Your answer:**

The model is able to show longer memory because output also depends on hidden state. Those hidden states are effected by previous batches 
and therfore since current model output is also affected by them it can produce out with longer memory then sequence len.

"""

part1_q3 = r"""
**Your answer:**
As mentioned before since each batch depends on hidden state created by previous batches, The order of batches is importent. 
Therfore shuffeling the batches will defect the content that is given from the corpus sentence order and the relation between consecutive batches which is expressed in hidden state.
"""

part1_q4 = r"""
**Your answer:**
1. The temperture hyper parameter help us to control the variance of distribution created by sofmax. High values
gives small variance and more uniform distribution. Therefore when we take temperture lower then 1.0 we can get more 
probability to higher scored predictions (in the training) and from then get less uniform. 
2. As we have seen in the graph above, when temperature increase we get distributdion that is closer to uniform. Threfore we are getting closer to random choose of next char because all choises have almost the same probality. So what we have learn has no effect and it will defect learning prcoess.
3. As we have seen in the graph above, when temperature over decrease our probablilty model created by the softmax will be degenerated and only the highest score predictions will be choosen since its probability is close to 1.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16, h_dim=128, z_dim=16, x_sigma2=0.001, learn_rate=0.0001, betas=(0.9, 0.99),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
$\sigma^2$ hyperparameter as implicit from its name control the variance of the output.
High values will give images that are more similar to each other, and low values will produce diversed images( different Boshes).
Also we can see from the loss function that $\sigma^2$ control how much weight we give to the reconstruction against the KL divergence.
"""

part2_q2 = r"""
**Your answer:**
1. Lets start with the reconstruction loss, its purpose to minimize the difference between the input and the output that was constructed by the encoder-decoder.
Now the second term the KL divergence is used to control the diversity of the outputs that produce by the model (As we will explain in 2).
2. The KL divergence is a regulization term that tries to minize how prior and post prior distributions are diffrent from each other and this way control the decoder output which is the latent space distribution model.
Therfore it controls how diversed are our sampels from latent space and how much it is close to normal distibution.

3. The benefit of this effect is that it tries to make the model more simple and this way we will avoid overfitting
(Which may occur in very complex function) and get better result in mapping output point to their latent space    coordinates.

"""

part2_q3 = r"""
**Your answer:**
First we have to remember that in this model we want to create output that is simillar to the input after reconstruction.
Therefore by maximizing P(X) we can get output distribution that is closer to the input image distribution in the data-set.

"""

part2_q4 = r"""
**Your answer:**
We are using log becuase it bring stability and ease of training. We know that sigma is very small number, therefore the optimaizer has to work with very small numbers whice cause to poorly gradient and numerical instabilities.
When using log if we take for exaple sigma in range [0,1] we map it to [-inf,0] and that gives us more space to work with.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,
        z_dim=128,
        data_label=0.0,
        label_noise=0.25,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0004,
            betas = (0.5,0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
             type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0004,
            betas = (0.5,0.9)
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
