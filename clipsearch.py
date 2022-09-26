
print('Please wait (loading can take around 15-30 seconds on ssd and gpu)')
print('1. importing')

import sys
import torch
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

print('2. loading models')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embbeding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs

def text_embbeding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs

print('Loading complete.')
print('Use (g) if you are using the tool first time.')
print('Use (l) if you want to restore previous session.')

option = input("(g)enerate embeddings and save, (l)oad from embeddings>")
match option:
    case "g":
        print('Enter path to directory with images (subdirectories will be included in the search)')
        print(r'Example: C:\Users\ghostdancing\Pictures')
        print('Tip: Use right click to paste')
        target = Path(input("Image folder path>"))
        print('It can take long time to process every image. 5 images per second on a modern gpu.')
        k_max = int(input("Maximum amount of images to process (ex. 1000)>"))

        print('Processing...')
        embeddings = []
        files = []
        k = 0

        all_paths = list(target.rglob("*.*"))
        print('Est. embedding size on disk =', len(all_paths) * 512 * 4 / 1000 / 1000, 'mb', len(all_paths))

        skipped = 0
        for full_path in tqdm(all_paths, total=min([len(all_paths), k_max])):

            if full_path.suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
                skipped += 1
                continue

            try:
                image = Image.open(full_path)
            except Exception as e:
                print(full_path)
                print(e)
                continue

            if image.width > 3000 and image.height > 3000:
                skipped += 1
                continue
            
            files.append(full_path.relative_to(target))

            embedding = image_embbeding(image)
            embeddings.append(embedding)

            k += 1
            if k > k_max: break

        print('Skipped images:', skipped)
        embeddings_tensor = torch.stack(embeddings).squeeze(1)
        print('Created embedding tensor:', embeddings_tensor.shape)
        
        torch.save(embeddings_tensor, 'clip_embeddings.pt')

        with open('clip_filenames.txt', 'w', encoding='utf8') as file:
            file.write (
                json.dumps({
                    'config': {
                        'target': str(target)
                    },
                    'files': [str(i) for i in files]
                })
            )
        
    case "l":
        embeddings_tensor = torch.load('clip_embeddings.pt')
        with open('clip_filenames.txt', 'r', encoding='utf8') as rfile:
            file = json.loads(rfile.read())

            target = Path(file['config']['target'])
            files = [Path(i) for i in file['files']]

        print('Loaded embedding tensor:', embeddings_tensor.shape)

    case _:
        print('Wrong option')
        sys.exit(0)

def find(query):
    search_query = text_embbeding(query)
    results = search_query @ embeddings_tensor.T

    results_sorted_indicies = results.argsort(descending=True).squeeze(0)
    
    sorted_filenames = [files[i] for i in results_sorted_indicies.tolist()]

    for i in results_sorted_indicies.tolist()[0:5]:
        print(F'Result #{i}:', results[0][i], results.shape)

    return sorted_filenames

plt.ion()
plt.close()

print('Seach mode. Type your query and a window will pop out with four images closely matching your query.')
print('For best results use full sentence, if there is ambiquity, provide more information.')
print('Example: A photo of a cat')
while True:
    query = input('query>')
    plt.close()
    fig = plt.figure()
    fig.suptitle(query)
    results = find('query = ' + query)

    for idx, i in enumerate(results[:4]):
        print(idx, i)
        ax = fig.add_subplot(2, 2, idx+1)
        plt.imshow(Image.open(target / i))

    plt.show()