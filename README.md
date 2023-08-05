# ImmersiveL
ImmersiveL is a framework and models hub for connecting languages all over the world open and free.

## Usage

Currently, you can use Deepspeed for preliminary translation tasks. You can modify the value of `input_sentences` to achieve translation effects.

The command to run is: 

```bash
deepspeed --num_gpus 8  bloomz_zero.py --name "bloomz-1b1" --batch_size 2
```

In the command, replace the --name parameter with your model, and replace --batch_size based on GPU memory.


## Setup
```bash
pip install torch
pip install transformers
pip install deepspeed
```
