## Title: Just Adjust One Prompt: Enhancing In-Context Dialogue Scoring via Constructing the Optimal Subgraph of Demonstrations and Prompts

## Overview

Our paper has been accpeted to EMNLP2023-main. This repository contains the code for our ADOROR method, where we plan to open source all the code with detailed
comments no later than **June 2024** (It may be ahead of schedule and we'll make it public as soon as we can!). We are very sorry, but for the time being we can only provide some of the code. Feel free to email me (iamlxb3@gmail.com) with any questions, my current research interest is doing dialog evaluation and persona dialog data generation based on in-context learning.

### Code for the core steps

- prepare_O.py: Code for Algorithm 1, how the data set O is constructed
- demp_graph.py: Code for Algorithm 2, how we conduct ensemble inference on the optimal subgraph
- demo_selector_vanilla_transformer.py: The model structure for f selector with vanilla transformer encoder
- demo_selector_rel_pos.py: The model structure for f selector with transformer encoder of relation position attention
  layer
