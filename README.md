F.R.I.E.N.D.S-GPT
=================

Description
-----------

F.R.I.E.N.D.S-GPT is a model designed to generate scripts based on the episodes of the TV show *Friends*. This project leverages the principles outlined in the "Attention is All You Need" paper, which introduces transformer models, a groundbreaking architecture for NLP-related tasks.

How It Works
------------

The generator focuses solely on producing scripts, implementing only the decoder part of the transformer model. Here's an overview of the approach:

1.  Data Preparation: The model is trained using a dataset of *Friends* scripts, which can be found [here](https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data).
2.  Model Architecture: Only the decoder part of the transformer model is used, as the generator doesn't require prompts or initial inputs. For tasks like language translation or query-response systems, the encoder part of the transformer would be needed.
3.  Training: The model is trained on script data to learn the patterns and structure of the dialogues.
4.  **Note** - This is a pretty basic representation of how such NLP tasks are accomplished. The final output generates valid text and words, not necessarily meaningful sentences. This obviously can be improved with a more indepth analysis and better hyper-parameter tuning.
   
Basic Transformer Architecture
------------------------------
![image](https://github.com/Noodle-bg/F.R.I.E.N.D.S-GPT/assets/142234652/6dd7e545-351d-4e89-8719-e4916f02db12)

* Only the Decoder section of the transformer is built in this project.
* This is because, the model is designed to just blindly generate scripts without any context of situation or prompt. So it doesn't need an encoder section for the model to build upon.


Results
-------

After training for 5000 epochs, the model achieved a train loss of 1.05 and a test loss of 1.16. These results can be improved further by adjusting hyperparameters and training for more epochs.

References
----------

-   Dataset: [Friends Netflix Script Data](https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data)
-   Paper: [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

This project represents my first venture into NLP, utilizing transformer models to generate character-based scripts. While this implementation might not be the only or the most accurate way to achieve the task, it provides a foundational understanding and can be built upon for further improvements.
Another really good watch on this topic is : https://www.youtube.com/watch?v=kCc8FmEb1nY . 
This video really helped me get a deeper and more proper understanding of how such transformers really work and how to convert them into code.
