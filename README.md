# From Zero to Hero
**Training a GAN from Scratch
For Image Inpainting**
##  Introduction
I wanted to learn as much as I can from this project, and to experiment with my model while implementing various ideas and techniques. I decided to try and train a GAN from
scratch. It allowed me to understand how every decision I make influence the
results, some for better and some for worse. My goal for this project was not to find the most cutting-edge solution, but to learn as much as possible. 
please see my [report](Image_inpainting___From_Zero_To_Hero.pdf) for more details.

##  Model Architecture
My model is composed of encoder-decoder generator architecture. Its
purpose is taking an image as input and down sampling it over a few layers
until a bottleneck layer, where the representation is then up sampled again over
a few layers before outputting the final image with the desired size. The encoder
is capturing the context of an image into a compact latent feature representation, and of Decoder architecture which uses that representation to produce the
missing image content. Between the encoder and the decoder, there is a channel2
wise fully connected layer, as described in the paper. I took inspiration from
the pix2pix  article described above and added a U-Net architecture between
the encoder and decoder. It was built using simple blocks, that was combined
together in a recursive manner. It will be further explained in the next section.
I implemented a global discriminator and PatchGAN discriminator. The global
discriminator looks at the whole image, while a local discriminator (PatchGAN
discriminator) looks at different regions of the filled image, which may give the
possibility to concentrate more on the masked regions.

## Results
![result1](results1.png)
![result2](results2.png)
![result3](results3.png)
