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
        seq_len=64,
        h_dim=1024,
        n_layers=3,
        dropout=0.3,
        learn_rate=0.001,
        lr_sched_factor=0.45,
        lr_sched_patience=3,
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
1. Since the corpus might be very large then including all data and RNN network information will not fit in memory. 
Therefore splitting into sequences will make the RNN network smaller and require less data avialble in RAM so it will prevent this issue.

2. This might help model not to over fit original corpus because these way each train 
    iteration that is unique, and in addition words in same sequence or batch have strong relation to each other
    because they are from same context in corpus so they create good sequential data to train iteration.
    These way we can prevent over-fit by training in each batch using serial sequences.
    
3. If we train on entire corpus it will cause our RNN network to be very deep. 
Therefore it might cause vanishing gradient issue that will make our model unattainable.


"""

part1_q2 = r"""
**Your answer:**

The model is able to show longer memory because output also depends on hidden state. Those hidden states are effected by previous batches 
and therefore since current model output is also affected by them it can produce out with longer memory then sequence len.

"""

part1_q3 = r"""
**Your answer:**
As mentioned before since each batch depends on hidden state created by previous batches, The order of batches is importent. 
Therefore shuffling the batches will defect the content that is given from the corpus sentence order and the relation 
between consecutive batches which is expressed in hidden state.
"""

part1_q4 = r"""
**Your answer:**
1. The temperature hyper parameter help us to control the variance of distribution created by sofmax. High values
gives small variance and more uniform distribution. Therefore when we take temperature lower then 1.0 we can get more get less uniform 
and more probability to higher scored predictions (in the training) that will reflect the result better. 

2. As we have seen in the graph above, when temperature increase we get distribution that is closer to uniform. 
Therefore we are getting closer to random choose of next char because all chooses have almost the same probability.
 So what we have learn has no effect and it will defect learning process.
 
3. As we have seen in the graph above, when temperature over decrease our probability model created by the softmax will 
be degenerated and only the highest score predictions will be chosen since its probability is close to 1 (deterministic).
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
        batch_size=16, h_dim=64, z_dim=32, x_sigma2=0.001, learn_rate=0.0001, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
$\sigma^2$ hyperparameter as implicit from its name control the variance of the output.
High values will give images that are more similar to dataset, and low values will produce more diverse with richer variety images( different Boshes).
Also we can see from the loss function that $\sigma^2$ control how much weight we give to the reconstruction against the KL divergence.
"""

part2_q2 = r"""
**Your answer:**
1. Lets start with the reconstruction loss, its purpose to minimize the difference between the input and the output that 
was constructed by the encoder-decoder.
Now the second term the KL divergence is used to control the diversity of the 
outputs that produce by the model by comparing latent output to wanted distribution (As we will explain in 2).

2. The KL divergence is a regularization term that tries to minimize how prior and post prior distributions are 
different from each other and this way control the decoder output which is the latent space distribution model.
Therefore it controls how diverse are our samples from latent space and how much it is close to normal distribution.

3. The benefit of this effect is that it tries to make the model more simple and this way we will avoid overfitting
(Which may occur in very complex function) and get better result in mapping output point to their latent space coordinates.


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
        z_dim=16,
        data_label=1,
        label_noise=0.25,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0003,
            betas = (0.5,0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
             type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0003,
            betas = (0.5,0.999)
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

We want to maintain the gradients only in the part when we train the generator.

We have two phases in our training process. First one to train the discriminator and second one to train the generator. 
In the training process our discriminator is using real image and image that was created by the generator, 
i.e. the generator is being used in both phases.
Therefore in first phase when we want to improve discriminator and update it's parameters according to loss function,
So we don't want to maintain the generator gradients and update it's parameters because we don't want it to be effected from
training process and errors of the discriminator, and vise versa.

In the second part when we try to improve the generator we want to maintain the generator gradients and update it's parameters.

To conclude we want to make sure training parameters of discriminator and generator change only in their training phase
although the discriminator is using the generator.
"""

part3_q2 = r"""
**Your answer:**

1. No. Since the loss function of generator score is based on how many images generated by him was labeled as fake 
   by the discriminator. Therefore if discriminator is bad and labels non realistic fake images as real ones then generator 
   loss will be very good regardless the images it generates, 
   i.e. generator loss is depended on the discriminator ability to recognize fake images.
   Therefore stopping training because the result of the generator loss is below a certain value is not a good idea 
   since it does not mean that the images produced are necessarily good as explained above. 

2. If the discriminator loss remains at a constant value while the generator loss decreases it might mean
    that generator learned faster then the discriminator, i.e. the generator is able to create fake images
    that the discriminator is not able to label as fake yet therefore it's loss is not improving and generator loss is
    improving.

"""

part3_q3 = r"""
**Your answer:**
In output images main difference between GAN and VAE is that VAE produced smoother images but blurrier images.
Reason for that is that the loss function is more simple and trained on real images that are smooth. 
Therefore generated images are less complex which makes the smoother.
On the other hand Gan model is much more complex (Based on generator and discriminator) , i.e. it 
can represent much more complex models/functions and therefore it can create much complex images
that might look more complex but less smooth and more noisy.

"""

# ==============
