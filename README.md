# CLIP cli search tool

Use text to search for any image in your photo/image folder.

Project uses openai's CLIP to create embedding of each image, and embedding of user's prompt.
This allows for quick one-shot search based on a few word description.

---
Dependencies:
- transformers
- pytorch
- matplotlib
- tqdm

tested on python>3.10