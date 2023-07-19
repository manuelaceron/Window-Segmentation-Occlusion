from datasets import load_dataset
import numpy as np
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from transformers import Mask2FormerImageProcessor
from torch.utils.data import DataLoader
import torch
from transformers import MaskFormerForInstanceSegmentation
import evaluate
from tqdm.auto import tqdm
import os

metric = evaluate.load("mean_iou")

ADE_MEAN = np.array([0.472455, 0.320782, 0.318403]) 
ADE_STD = np.array([0.215084, 0.408135, 0.409993])

train_transform = A.Compose([
    A.Flip(p=0.5),
    #aug.RandomCrop(width=128, height=128), does not work for this dataset and task
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    A.geometric.rotate.Rotate (limit=[-15, 15])
])

test_transform = A.Compose([
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])


def transforms(examples):
    examples["label"] = [image.convert("L") for image in examples["label"]]

    threshold = 127  # Adjust the threshold as needed
     
    examples["label"] = [np.array(image) > threshold for image in examples["label"]]
    examples["label"] = [image.astype(np.uint8)*255 for image in examples["label"]]
    examples["label"] = [Image.fromarray(image) for image in examples["label"]]

    return examples

def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )
    
    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]

    return batch

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]['image'])
        original_segmentation_map = np.array(self.dataset[idx]['label'])
        original_segmentation_map = np.where(original_segmentation_map>0, 1, 0)

        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2,0,1)

        return image, segmentation_map, original_image, original_segmentation_map

dataset = load_dataset("manuCeron96/FacadeDataset")
new_dataset = dataset.map(transforms, batched=True)

train_ds = new_dataset["train"]
test_ds = new_dataset["test"]
val_ds = new_dataset["validation"]

# Create dataset
train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
val_dataset = ImageSegmentationDataset(val_ds, transform=train_transform)
test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)

# Create a preprocessor
preprocessor = Mask2FormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Check data
batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)


# Define model
id2label = { 0: "background", 1: "window"}
label2id = { "background": 0, "window": 1 }

# Replace the head of the pre-trained model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade", id2label=id2label, ignore_mismatched_sizes=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
epochs = 100

tpbar = tqdm(train_dataloader)

for epoch in range(epochs):
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tpbar):
      # Reset the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(
          pixel_values=batch["pixel_values"].to(device),
          mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
          class_labels=[labels.to(device) for labels in batch["class_labels"]],
      )

      # Backward propagation
      loss = outputs.loss
      loss.backward()

      batch_size = batch["pixel_values"].size(0)
      running_loss += loss.item()
      num_samples += batch_size

      if idx % 100 == 0:
        print("Loss:", running_loss/num_samples)

      # Optimization
      optimizer.step()

      original_images = batch["original_images"]
      target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
      predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)

      ground_truth_segmentation_maps = batch["original_segmentation_maps"]
      metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
      tpbar.set_postfix({'IoU':metric.compute(num_labels = len(id2label), ignore_index = 0)['per_category_iou']})

  model.eval()
  pbar = tqdm(val_dataloader)
  for idx, batch in enumerate(pbar):
    #if idx > 5:
    #  break

    pixel_values = batch["pixel_values"]

    # Forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values.to(device))

    # get original images
    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)

    # get ground truth segmentation maps
    ground_truth_segmentation_maps = batch["original_segmentation_maps"]

    metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)

  print("Mean IoU:", metric.compute(num_labels = len(id2label), ignore_index = 0)['per_category_iou'])

  checkpoint = {
      "net": model.state_dict(),
      'optimizer': optimizer.state_dict(),
      "epoch": epoch
      }
  name = str(epoch)+'model.pth'
  torch.save(checkpoint, os.path.join("/home/cero_ma/MCV/code220419_windows/0401_files/Mask2Former_ecp/", name))